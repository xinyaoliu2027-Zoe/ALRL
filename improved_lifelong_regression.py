import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from collections import OrderedDict
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import os
import warnings

warnings.filterwarnings('ignore')


class InputAdapter(nn.Module):
    """Input adapter: Maps variable-dimension inputs to fixed-size representations"""

    def __init__(self, input_dim, output_dim=128):
        super(InputAdapter, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Use multi-layer perceptron as adapter
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)  # Use layer normalization instead of batch norm to avoid single sample issues
        )

    def forward(self, x):
        return self.network(x)


class SharedBackbone(nn.Module):
    """Globally shared backbone (single instance per model).

    Operates on the 128-dimensional input-adapted representation z_t and
    produces a 256-dimensional feature h_t shared across all tasks. This
    is the parameter set that is regularised by selective EWC and that
    catastrophic forgetting can actually act on, because every task's
    forward pass goes through the *same* backbone object.
    """

    def __init__(self, in_dim=128, out_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.layers(x)


class DynamicNetwork(nn.Module):
    """Task-specific head.

    Note: despite the historical class name, this module no longer
    contains any shared layers — the shared backbone is now a separate,
    globally shared object owned by AdvancedLifelongRegressionModel
    (class SharedBackbone). This module only implements the
    task-specific transformation from h_t to the scalar prediction.
    """

    def __init__(self, in_dim=256, hidden_dim=128, output_dim=1):
        super().__init__()
        self.task_layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.task_layers(x)

    def get_task_params(self):
        return {name: param for name, param in self.named_parameters()
                if 'task_layers' in name}


class TaskEmbeddingNetwork(nn.Module):
    """Task embedding network: Generate low-dimensional task representations for similarity detection"""

    def __init__(self, input_dim, embedding_dim=64):
        super(TaskEmbeddingNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.Tanh()  # Use tanh to normalize embedding vectors for stable comparison
        )

    def forward(self, x):
        return self.network(x)


class AdvancedLifelongRegressionModel:
    """Advanced lifelong regression learning model that can handle different input dimensions"""

    def __init__(
            self,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            buffer_size=5000,
            shared_dim=128,  # Input adapter output dimension (unified shared representation)
            embedding_dim=64,
            similarity_threshold=0.25,  # Lower similarity threshold to more easily find similar tasks
            visualization_dir='visualizations'
    ):
        self.device = device
        self.task_adapters = {}  # Task-specific input adapters
        self.task_networks = {}  # Task-specific main networks
        self.task_embeddings = {}  # Task embedding networks
        self.task_importances = {}  # Task parameter importance
        self.old_params = {}  # Old parameter storage

        # Configuration parameters
        self.shared_dim = shared_dim
        self.embedding_dim = embedding_dim
        self.buffer_size = buffer_size
        self.similarity_threshold = similarity_threshold
        self.visualization_dir = visualization_dir

        # Create visualization directory
        os.makedirs(visualization_dir, exist_ok=True)

        # Memory buffer - use OrderedDict to track sample insertion order
        self.memory_buffer = OrderedDict()
        self.buffer_counter = 0  # Used for generating unique keys

        # Task information storage
        self.task_datasets = {}
        self.task_dims = {}
        self.task_metrics = {}

        # Similarity records
        self.task_similarities = {}

        # Add ID mapping dictionary - for resolving task ID format inconsistency issues
        self.task_id_mapping = {}  # Original ID -> Internal ID
        self.reverse_id_mapping = {}  # Internal ID -> Original ID

        # ------------------------------------------------------------------
        # Globally-shared backbone. Created ONCE per model and shared
        # across all tasks. This is what makes catastrophic forgetting
        # possible (and EWC meaningful) — every task's forward pass runs
        # through this same object, so training on a new task can drift
        # the parameters that earlier tasks rely on.
        # ------------------------------------------------------------------
        self.shared_backbone = SharedBackbone(in_dim=shared_dim, out_dim=256).to(device)
        # Snapshot of shared-backbone params after each task finishes,
        # used as anchors by EWC.
        self.old_shared_params = {}     # task_id -> {param_name: tensor}
        self.shared_importances = {}    # task_id -> {param_name: tensor}

        print(f"Initializing model: device={device}, shared_dim={shared_dim}, embedding_dim={embedding_dim}")

    def _prepare_data(self, X, y, batch_size=32, shuffle=True):
        """Prepare data loader"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _compute_statistics(self, X):
        """Compute dataset statistics for feature representation"""
        stats = {}

        # Basic statistics
        stats['mean'] = np.mean(X, axis=0)
        stats['std'] = np.std(X, axis=0) + 1e-10  # Add small constant to avoid division by zero
        stats['min'] = np.min(X, axis=0)
        stats['max'] = np.max(X, axis=0)
        stats['range'] = stats['max'] - stats['min']

        # Higher-order statistics
        stats['skew'] = np.mean((X - stats['mean']) ** 3, axis=0) / (stats['std'] ** 3)
        stats['kurtosis'] = np.mean((X - stats['mean']) ** 4, axis=0) / (stats['std'] ** 4)

        # Correlation statistics
        if X.shape[1] > 1:
            corr_matrix = np.corrcoef(X.T)
            stats['corr_mean'] = np.mean(np.abs(corr_matrix[np.triu_indices(X.shape[1], k=1)]))
            stats['corr_std'] = np.std(np.abs(corr_matrix[np.triu_indices(X.shape[1], k=1)]))
        else:
            stats['corr_mean'] = 0
            stats['corr_std'] = 0

        # Dimension information
        stats['dim'] = X.shape[1]
        stats['sample_count'] = X.shape[0]

        return stats

    def _extract_feature_signatures(self, X):
        """Extract signature features from dataset for better similarity detection"""
        # Sample a subset for efficiency if dataset is large
        if X.shape[0] > 1000:
            indices = np.random.choice(X.shape[0], 1000, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        # Calculate feature-level statistics
        feature_sigs = []
        for i in range(X.shape[1]):
            feature = X_sample[:, i]

            # Distribution statistics
            q25, q50, q75 = np.percentile(feature, [25, 50, 75])

            # Calculate density at certain points (approximation of PDF)
            hist, bins = np.histogram(feature, bins=10, density=True)

            # Create feature signature
            sig = {
                'mean': np.mean(feature),
                'std': np.std(feature),
                'min': np.min(feature),
                'max': np.max(feature),
                'q25': q25,
                'median': q50,
                'q75': q75,
                'skew': (q75 + q25 - 2 * q50) / (q75 - q25) if q75 > q25 else 0,  # Simplified skew
                'hist': hist.tolist()
            }
            feature_sigs.append(sig)

        return feature_sigs

    def _compare_feature_signatures(self, sig1, sig2):
        """Compare two feature signatures to determine similarity"""
        # Handle different dimensions by comparing only the number of features that both have
        min_features = min(len(sig1), len(sig2))

        similarities = []
        for i in range(min_features):
            # Compare statistical moments
            moment_sim = 1.0 - 0.25 * (
                    abs(sig1[i]['mean'] - sig2[i]['mean']) +
                    abs(sig1[i]['std'] - sig2[i]['std']) +
                    abs(sig1[i]['skew'] - sig2[i]['skew']) +
                    abs((sig1[i]['q75'] - sig1[i]['q25']) - (sig2[i]['q75'] - sig2[i]['q25']))
            )

            # Compare histograms using a simple correlation
            hist1 = np.array(sig1[i]['hist'])
            hist2 = np.array(sig2[i]['hist'])

            # Normalize histograms
            hist1 = hist1 / (np.sum(hist1) + 1e-10)
            hist2 = hist2 / (np.sum(hist2) + 1e-10)

            # Calculate correlation
            hist_sim = np.corrcoef(hist1, hist2)[0, 1] if not np.isnan(np.corrcoef(hist1, hist2)[0, 1]) else 0

            # Combined similarity
            feature_sim = 0.7 * moment_sim + 0.3 * hist_sim
            similarities.append(feature_sim)

        # When tasks have at least 5 input features, emphasize the leading
        # three columns (which carry the strongest statistical descriptors
        # under our standardization pipeline) by a 1.5x weight; otherwise
        # fall back to a uniform average over the available features.
        if min_features >= 5:
            weights = np.ones(min_features)
            weights[:3] *= 1.5
            weights = weights / np.sum(weights)
            return np.sum(similarities * weights)
        else:
            return np.mean(similarities)

    def _get_task_embedding_input(self, X, y):
        """Get task embedding input from dataset (dimension-independent)"""
        X_stats = self._compute_statistics(X)
        y_stats = {'mean': np.mean(y), 'std': np.std(y), 'min': np.min(y), 'max': np.max(y)}

        # Feature distribution
        feature_dist = []
        for i in range(min(X.shape[1], 5)):  # Sample max 5 features to keep dimension fixed
            if i < X.shape[1]:
                # Get distribution characteristics of this feature
                feature = X[:, i]
                q25, q50, q75 = np.percentile(feature, [25, 50, 75])
                feature_dist.extend([q25, q50, q75])
            else:
                # Padding for smaller dimensions
                feature_dist.extend([0, 0, 0])

        # Target distribution characteristics
        y_q25, y_q50, y_q75 = np.percentile(y, [25, 50, 75])

        # Aggregate statistical features to create fixed-dimension task representation
        embedding_features = [
            # Basic dataset properties (normalized)
            np.log1p(X.shape[0]) / 10,  # Sample count (log-scaled)
            np.log1p(X.shape[1]) / 5,  # Feature count (log-scaled)

            # Feature statistics
            np.mean(X_stats['mean']),
            np.std(X_stats['mean']),
            np.mean(X_stats['std']),
            np.std(X_stats['std']),
            np.mean(X_stats['skew']),
            np.std(X_stats['skew']),

            # Correlation structure
            X_stats['corr_mean'],
            X_stats['corr_std'],

            # Target statistics
            y_stats['mean'],
            y_stats['std'],
            (y_stats['max'] - y_stats['min']),
            y_q25, y_q50, y_q75,

            # X-Y relationship
            np.corrcoef(np.mean(X, axis=1), y)[0, 1] if X.shape[0] > 1 else 0,
        ]

        # Add sampled feature distributions
        embedding_features.extend(feature_dist)

        # Convert to numpy array
        embedding_input = np.array(embedding_features, dtype=np.float32)

        return embedding_input

    def _calculate_task_similarity(self, task_id, X, y):
        """Calculate similarity between new task and existing tasks (dimension-independent)"""
        if not self.task_embeddings:
            return {}

        # Extract feature signatures for the new task
        new_feature_sigs = self._extract_feature_signatures(X)

        # Generate embedding input for new task
        new_embedding_input = self._get_task_embedding_input(X, y)

        # Create temporary embedding network
        temp_embedding_net = TaskEmbeddingNetwork(
            input_dim=len(new_embedding_input),
            embedding_dim=self.embedding_dim
        ).to(self.device)

        # Randomly initialize temporary embedding network
        with torch.no_grad():
            new_embedding = temp_embedding_net(
                torch.FloatTensor(new_embedding_input).to(self.device)
            )

        # Calculate cosine similarity with all existing tasks
        similarities = {}
        for internal_tid, embedding_net in self.task_embeddings.items():
            with torch.no_grad():
                # Get existing task embedding
                original_tid = self.reverse_id_mapping.get(internal_tid, internal_tid)

                # Get dataset for the existing task
                task_X = self.task_datasets[internal_tid]['X']
                task_y = self.task_datasets[internal_tid]['y']

                # Extract feature signatures for the existing task
                task_feature_sigs = self._extract_feature_signatures(task_X)

                # Calculate feature-based similarity
                feature_similarity = self._compare_feature_signatures(new_feature_sigs, task_feature_sigs)

                # Get embedding-based similarity
                task_embedding_input = self._get_task_embedding_input(task_X, task_y)
                task_embedding = embedding_net(
                    torch.FloatTensor(task_embedding_input).to(self.device)
                )
                embedding_similarity = torch.nn.functional.cosine_similarity(
                    new_embedding, task_embedding, dim=0
                ).item()

                # Combine the closed-form signature similarity and the
                # embedding cosine similarity with equal weights.
                combined_similarity = 0.5 * feature_similarity + 0.5 * embedding_similarity

            # Store similarity using original ID
            similarities[original_tid] = combined_similarity

        # Save similarity record (using original ID)
        self.task_similarities[task_id] = similarities

        return similarities

    def _add_to_buffer(self, task_id, X, y, importance=1.0, max_samples=200):
        """Add samples to memory buffer"""
        # Randomly select samples
        indices = np.random.permutation(len(X))[:max_samples]

        # Add to buffer
        for idx in indices:
            # Use incrementing counter as key to ensure uniqueness
            key = f"{self.buffer_counter}"
            self.buffer_counter += 1

            # Ensure X[idx] is a one-dimensional array
            x_sample = X[idx].flatten() if isinstance(X[idx], np.ndarray) else X[idx]

            self.memory_buffer[key] = {
                'x': x_sample,
                'y': y[idx],
                'task_id': task_id,
                'dim': X.shape[1],
                'importance': importance,
                'time_added': time.time()  # Add timestamp for aging policy
            }

            # If buffer size exceeded, remove oldest sample
            if len(self.memory_buffer) > self.buffer_size:
                oldest_key = next(iter(self.memory_buffer))
                del self.memory_buffer[oldest_key]

    def _process_memory_batch(self, batch_items, target_dim, adapter, network, criterion):
        """Process memory batch samples"""
        try:
            x_list = []
            y_list = []

            # Preprocess all samples
            for item in batch_items:
                x = item['x']
                y = item['y']

                # Ensure x has correct shape
                if isinstance(x, list) and len(x) == target_dim:
                    x_list.append(x)
                elif isinstance(x, np.ndarray) and x.size == target_dim:
                    x_list.append(x.reshape(target_dim))
                else:
                    continue  # Skip incompatible samples

                y_list.append(y)

            if not x_list:  # If no valid samples
                return None

            # Convert lists to tensors - first convert to numpy arrays
            x_tensor = torch.FloatTensor(np.array(x_list)).to(self.device)
            y_tensor = torch.FloatTensor(y_list).view(-1, 1).to(self.device)

            # Forward: replayed-task adapter -> SHARED backbone -> current task head
            adapted_x = adapter(x_tensor)
            shared_feat = self.shared_backbone(adapted_x)
            mem_output = network(shared_feat)
            mem_loss = criterion(mem_output, y_tensor)

            return mem_loss

        except Exception as e:
            print(f"Memory batch processing error: {e}")
            return None

    def _sample_from_buffer(self, batch_size=32, target_dim=None, target_task=None):
        """Sample from memory buffer, grouped by task and dimension"""
        if len(self.memory_buffer) < batch_size // 2:  # Ensure enough samples
            return None, None, None, None

        # Calculate sampling weights
        keys = list(self.memory_buffer.keys())
        weights = []

        current_time = time.time()
        for key in keys:
            item = self.memory_buffer[key]
            weight = item['importance']

            # If target dimension specified, increase weight of matching dimension samples
            if target_dim is not None and item['dim'] == target_dim:
                weight *= 2.0

            # If target task specified, increase weight of that task's samples
            if target_task is not None:
                if item['task_id'] == target_task:
                    weight *= 1.5
                elif target_task in self.task_similarities and item['task_id'] in self.task_similarities[target_task]:
                    # Adjust weight based on task similarity
                    sim = self.task_similarities[target_task][item['task_id']]
                    weight *= (1.0 + max(0, sim))

            # Add time decay factor (newer samples have higher weight)
            time_factor = np.exp(-0.1 * (current_time - item['time_added']) / (3600 * 24))  # Decay ~10% per day
            weight *= max(0.5, time_factor)  # Keep at least 50% of base weight

            weights.append(weight)

        # Normalize weights
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(keys)) / len(keys)  # If all weights zero, use uniform distribution

        # Sample by weight
        sampled_indices = np.random.choice(
            len(keys),
            size=min(batch_size, len(keys)),
            replace=False,
            p=weights
        )

        # Group by task and dimension
        memory_batches = {}

        for idx in sampled_indices:
            key = keys[idx]
            item = self.memory_buffer[key]

            # Create batch key
            batch_key = f"{item['task_id']}_{item['dim']}"
            if batch_key not in memory_batches:
                memory_batches[batch_key] = []

            memory_batches[batch_key].append(item)

        return memory_batches, None, None, None  # Only return grouped batches

    def _compute_importance(self, task_id, X, y, samples=100):
        """Compute Fisher-diagonal importance for the SHARED backbone only.

        With the new architecture, the only parameters that experience
        cross-task interference are those of self.shared_backbone (the
        per-task adapter and per-task head are not modified by future
        tasks). We therefore estimate Fisher information only for the
        shared backbone, store both the importances and a snapshot of
        the post-task backbone parameters, and use them as anchors in
        _apply_ewc_regularization() during subsequent tasks.
        """
        adapter = self.task_adapters[task_id]
        task_head = self.task_networks[task_id]

        adapter.eval()
        task_head.eval()
        self.shared_backbone.eval()

        # Importance and snapshot dicts (shared-backbone params only).
        importance = {name: torch.zeros_like(p)
                      for name, p in self.shared_backbone.named_parameters()}
        snapshot = {name: p.clone().detach()
                    for name, p in self.shared_backbone.named_parameters()}

        self.shared_importances[task_id] = importance
        self.old_shared_params[task_id] = snapshot

        indices = np.random.choice(len(X), min(samples, len(X)), replace=False)
        criterion = nn.MSELoss()

        for idx in indices:
            x = torch.FloatTensor(X[idx:idx + 1]).to(self.device)
            target = torch.FloatTensor([y[idx]]).reshape(-1, 1).to(self.device)

            # Forward & backward to populate gradients on the shared backbone.
            adapter.zero_grad()
            task_head.zero_grad()
            self.shared_backbone.zero_grad()

            shared_feat = self.shared_backbone(adapter(x))
            output = task_head(shared_feat)
            loss = criterion(output, target)
            loss.backward()

            for name, p in self.shared_backbone.named_parameters():
                if p.grad is not None:
                    importance[name] += p.grad.data ** 2

        for name in importance:
            importance[name] /= len(indices)

        # Also keep a copy in self.task_importances so external code that
        # introspects this dict (e.g. analyze_*) does not break.
        self.task_importances[task_id] = importance
        self.old_params[task_id] = snapshot
        return importance

    def _apply_ewc_regularization(self, task_id, similar_tasks, loss,
                                  current_adapter, current_network, ewc_lambda):
        """Selective EWC: anchor the shared backbone to its post-task
        snapshots from the *similar* past tasks only.

        L_EWC = (1/2) * sum_{k in S_t} sum_i F_{k,i} (theta_i - theta_{k,i}^*)^2

        Only parameters of self.shared_backbone are penalised — adapters
        and task heads are not (they are not updated when learning a
        future task).
        """
        if not similar_tasks or ewc_lambda <= 0:
            return loss

        ewc_loss = 0.0
        live = {name: p for name, p in self.shared_backbone.named_parameters()}

        for sim_task_id in similar_tasks:
            if sim_task_id not in self.shared_importances:
                continue
            imp = self.shared_importances[sim_task_id]
            old = self.old_shared_params[sim_task_id]

            for name, F in imp.items():
                if name not in old or name not in live:
                    continue
                if old[name].shape != live[name].shape:
                    # Shouldn't happen with a globally shared backbone, but be defensive.
                    continue
                ewc_loss = ewc_loss + (F * (live[name] - old[name]) ** 2).sum()

        if isinstance(ewc_loss, torch.Tensor) and ewc_loss.requires_grad:
            return loss + 0.5 * ewc_lambda * ewc_loss
        return loss

    def _create_task_models(self, task_id, input_dim, similar_task_id=None):
        """Create per-task modules.

        With the new architecture, there is exactly ONE shared backbone
        per model (self.shared_backbone), so this function only ever
        instantiates per-task pieces:
          * a fresh input adapter A_t  (R^{d_t} -> R^{128})
          * a task head G_t            (R^{256} -> R^{1})
          * a fresh task-embedding network g_psi_t

        If a similar past task exists, the new task head is initialised
        by deep-copying that task's head (similarity-guided warm start).
        The shared backbone is *never* deep-copied — it is the single
        global object that catastrophic forgetting can act on.
        """
        # Per-task input adapter (always new because input dim varies).
        adapter = InputAdapter(input_dim, self.shared_dim).to(self.device)

        # Per-task head: warm-start from the most similar past task if any.
        if similar_task_id is not None and similar_task_id in self.task_networks:
            print(f"Warm-starting task head from similar task {similar_task_id}")
            task_head = copy.deepcopy(self.task_networks[similar_task_id])
        else:
            print(f"Creating new task head for task {task_id}")
            task_head = DynamicNetwork(in_dim=256).to(self.device)

        # Per-task embedding projection.
        embedding_input = self._get_task_embedding_input(
            self.task_datasets[task_id]['X'], self.task_datasets[task_id]['y']
        )
        embedding_net = TaskEmbeddingNetwork(
            len(embedding_input), self.embedding_dim
        ).to(self.device)

        return adapter, task_head, embedding_net

    def _get_similar_tasks(self, task_id, similarities, threshold=None):
        """Get list of similar tasks"""
        if threshold is None:
            threshold = self.similarity_threshold

        similar_tasks = []
        for tid, similarity in similarities.items():
            if similarity > threshold:
                # Get internal task ID
                internal_tid = self.task_id_mapping.get(tid, tid)
                similar_tasks.append(internal_tid)

        return similar_tasks

    def train_task(self, task_id, X, y, epochs=100, batch_size=32, lr=0.001, ewc_lambda=100, similarity_threshold=None):
        """Train model for specific task"""
        # Save original task ID
        original_task_id = task_id

        # Create internal task ID (replace underscores with hyphens)
        internal_task_id = str(task_id).replace('_', '-')

        # Update ID mapping
        self.task_id_mapping[original_task_id] = internal_task_id
        self.reverse_id_mapping[internal_task_id] = original_task_id

        # Use internal ID for training
        task_id = internal_task_id

        print(f"\nStarting training for task: {original_task_id}, input dimension: {X.shape[1]}")

        # Store task dataset information
        self.task_datasets[task_id] = {'X': X, 'y': y}
        self.task_dims[task_id] = X.shape[1]

        # Calculate similarity with existing tasks
        similarities = self._calculate_task_similarity(original_task_id, X, y)

        # Print similarity information
        if similarities:
            print("Task similarities:")
            for tid, sim in similarities.items():
                print(f"  Similarity with task {tid}: {sim:.4f}")

        # Get similar tasks
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
        similar_tasks = self._get_similar_tasks(original_task_id, similarities, similarity_threshold)
        most_similar_task = None
        if similar_tasks:
            # Find most similar task
            most_similar_task = max(similar_tasks, key=lambda t: similarities[self.reverse_id_mapping.get(t, t)])

            # Print using original ID
            most_similar_task_orig = self.reverse_id_mapping.get(most_similar_task, most_similar_task)
            print(
                f"Found most similar task: {most_similar_task_orig}, similarity: {similarities[most_similar_task_orig]:.4f}")

        # Create task models
        adapter, network, embedding_net = self._create_task_models(
            task_id, X.shape[1], most_similar_task
        )

        # Save models
        self.task_adapters[task_id] = adapter
        self.task_networks[task_id] = network
        self.task_embeddings[task_id] = embedding_net

        # Learning rate adjustment: use smaller learning rate for the
        # GLOBALLY shared backbone when warm-starting from a similar
        # task (this protects knowledge from earlier tasks); keep full
        # learning rate for the task-specific head and adapter.
        shared_lr = lr * 0.2 if most_similar_task else lr

        # Set optimizer with layer-specific learning rates. Note that
        # self.shared_backbone is the SINGLE global backbone shared
        # across all tasks; updates here will affect every task's
        # forward pass on subsequent evaluations, which is what makes
        # catastrophic forgetting (and EWC) meaningful.
        optimizer = optim.Adam([
            {'params': adapter.parameters(), 'lr': lr},
            {'params': self.shared_backbone.parameters(), 'lr': shared_lr},
            {'params': network.task_layers.parameters(), 'lr': lr},
            {'params': embedding_net.parameters(), 'lr': lr},
        ])

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.5, patience=5
        )

        # Prepare data
        data_loader = self._prepare_data(X, y, batch_size)
        criterion = nn.MSELoss()

        # Training loop
        best_loss = float('inf')
        patience = 10  # Early stopping patience
        patience_counter = 0

        for epoch in range(epochs):
            running_loss = 0.0

            for X_batch, y_batch in data_loader:
                # Forward pass through:  adapter -> shared backbone -> task head
                adapter.train()
                self.shared_backbone.train()
                network.train()

                shared_feat = self.shared_backbone(adapter(X_batch))
                outputs = network(shared_feat)

                # Calculate loss
                loss = criterion(outputs, y_batch.view(-1, 1))

                # Add memory replay
                if epoch > 0 and len(self.memory_buffer) > batch_size // 2:
                    memory_batches, _, _, _ = self._sample_from_buffer(
                        batch_size, target_task=task_id
                    )

                    if memory_batches:
                        # Process each batch
                        memory_loss = 0.0
                        memory_count = 0

                        for batch_key, batch_items in memory_batches.items():
                            # Fix parsing logic - ensure correct delimiter and error handling
                            try:
                                parts = batch_key.split('_')
                                if len(parts) != 2:
                                    print(f"Warning: Skipping improperly formatted batch key: {batch_key}")
                                    continue

                                mem_task_id, mem_dim_str = parts
                                mem_dim = int(mem_dim_str)

                                # Select appropriate adapter
                                if mem_dim != X.shape[1] and mem_task_id in self.task_adapters:
                                    # Use original task's adapter
                                    mem_adapter = self.task_adapters[mem_task_id]
                                else:
                                    # Use current adapter
                                    mem_adapter = adapter

                                batch_loss = self._process_memory_batch(batch_items, mem_dim, mem_adapter, network,
                                                                        criterion)

                                if batch_loss is not None:
                                    memory_loss += batch_loss
                                    memory_count += 1
                            except Exception as e:
                                print(f"Error processing memory batch: {e}, key={batch_key}")
                                continue

                        if memory_count > 0:
                            # Adjust replay weight based on training progress (reduce later)
                            memory_weight = 0.3 * (1.0 - epoch / epochs)
                            loss += memory_weight * (memory_loss / memory_count)

                # Add EWC regularization
                if similar_tasks and ewc_lambda > 0:
                    loss = self._apply_ewc_regularization(
                        task_id, similar_tasks, loss, adapter, network, ewc_lambda
                    )

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)

                optimizer.step()

                running_loss += loss.item()

                # End of epoch, calculate average loss
            avg_loss = running_loss / len(data_loader)

            # Learning rate scheduling
            scheduler.step(avg_loss)

            # Print training progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Task {original_task_id}, Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

                # Evaluate on validation set
                if epoch > 0 and epoch % 20 == 0:
                    val_indices = np.random.choice(len(X), min(100, len(X)), replace=False)
                    val_metrics = self.evaluate(original_task_id, X[val_indices], y[val_indices])
                    print(f"  Validation performance: R²={val_metrics['r2']:.4f}, MSE={val_metrics['mse']:.4f}")

            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

            # Compute parameter importance
        print("Computing parameter importance...")
        importance = self._compute_importance(task_id, X, y)
        self.task_importances[task_id] = importance

        # Add samples to memory buffer
        print("Adding samples to memory buffer...")
        self._add_to_buffer(task_id, X, y, importance=1.0)

        # Evaluate and save final performance
        print("Evaluating final performance...")
        val_indices = np.random.choice(len(X), min(200, len(X)), replace=False)
        final_metrics = self.evaluate(original_task_id, X[val_indices], y[val_indices])
        self.task_metrics[original_task_id] = final_metrics

        print(f"Task {original_task_id} training completed")
        print(
            f"Final performance: R²={final_metrics['r2']:.4f}, MSE={final_metrics['mse']:.4f}, MAE={final_metrics['mae']:.4f}")

        return final_metrics

    def predict(self, task_id, X):
        """Make predictions for a given task.

        Forward path: task-specific adapter A_t  ->  SHARED backbone  ->
                       task-specific head G_t  ->  scalar y_hat.

        Crucially, self.shared_backbone is the single global object that
        is mutated by every subsequent training task. Re-evaluating an
        earlier task therefore reads the *current* (potentially drifted)
        backbone, which is what makes catastrophic forgetting visible.
        """
        internal_task_id = self.task_id_mapping.get(task_id, task_id)

        if internal_task_id not in self.task_adapters or internal_task_id not in self.task_networks:
            raise ValueError(f"Task {task_id} has not been trained")

        adapter = self.task_adapters[internal_task_id]
        task_head = self.task_networks[internal_task_id]

        adapter.eval()
        self.shared_backbone.eval()
        task_head.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            shared_feat = self.shared_backbone(adapter(X_tensor))
            predictions = task_head(shared_feat)

        return predictions.cpu().numpy()

    def evaluate(self, task_id, X, y):
        """Evaluate model performance on specific task"""
        predictions = self.predict(task_id, X)

        if len(predictions.shape) > 1 and predictions.shape[1] == 1:
            predictions = predictions.squeeze()

        # Calculate evaluation metrics
        metrics = {
            'r2': r2_score(y, predictions),
            'mse': mean_squared_error(y, predictions),
            'mae': mean_absolute_error(y, predictions)
        }

        return metrics

    def visualize_task_similarities(self):
        """Visualize task similarity matrix"""
        if not self.task_similarities:
            print("No task similarity data available for visualization")
            return

        # Use original task IDs for visualization
        task_ids = list(self.task_metrics.keys())
        n_tasks = len(task_ids)

        # Create similarity matrix
        sim_matrix = np.zeros((n_tasks, n_tasks))

        for i, task1 in enumerate(task_ids):
            for j, task2 in enumerate(task_ids):
                if task1 == task2:
                    sim_matrix[i, j] = 1.0
                elif task1 in self.task_similarities and task2 in self.task_similarities[task1]:
                    sim_matrix[i, j] = self.task_similarities[task1][task2]
                elif task2 in self.task_similarities and task1 in self.task_similarities[task2]:
                    sim_matrix[i, j] = self.task_similarities[task2][task1]

        # Set matplotlib font to support Unicode
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'Bitstream Vera Sans',
                                           'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(sim_matrix, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='Similarity')

        # Add labels
        plt.xticks(range(n_tasks), task_ids, rotation=45)
        plt.yticks(range(n_tasks), task_ids)

        # Add value labels
        for i in range(n_tasks):
            for j in range(n_tasks):
                plt.text(j, i, f"{sim_matrix[i, j]:.2f}", ha='center', va='center',
                         color='white' if sim_matrix[i, j] < 0.7 else 'black')

        plt.title('Task Similarity Matrix')
        plt.tight_layout()

        # Save matrix data
        os.makedirs(self.visualization_dir, exist_ok=True)
        filename = os.path.join(self.visualization_dir, 'task_similarity_matrix.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Task similarity matrix saved to {filename}")

    def visualize_forgetting(self):
        """Visualize catastrophic forgetting patterns"""
        if not hasattr(self, 'forgetting_metrics'):
            print("No forgetting metrics data available for visualization")
            return

        task_ids = list(self.forgetting_metrics.keys())
        metrics = ['r2', 'mse', 'mae']
        metric_labels = {'r2': 'R²', 'mse': 'MSE', 'mae': 'MAE'}

        # Set matplotlib font to support Unicode
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'Bitstream Vera Sans',
                                           'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, metric in enumerate(metrics):
            for task_id in task_ids:
                task_metrics = self.forgetting_metrics[task_id][metric]
                epochs = range(len(task_metrics))
                axes[i].plot(epochs, task_metrics, marker='o', label=f'Task {task_id}')

            axes[i].set_title(f'{metric_labels[metric]} Over Time')
            axes[i].set_xlabel('Task Sequence')
            axes[i].set_ylabel(metric_labels[metric])
            axes[i].grid(True)
            axes[i].legend()

        plt.tight_layout()

        # Save figure
        os.makedirs(self.visualization_dir, exist_ok=True)
        filename = os.path.join(self.visualization_dir, 'forgetting_patterns.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Forgetting patterns saved to {filename}")

    def visualize_performance(self):
        """Visualize final model performance across tasks"""
        if not self.task_metrics:
            print("No performance data available for visualization")
            return

        # Extract performance metrics
        task_ids = list(self.task_metrics.keys())
        r2_scores = [self.task_metrics[tid]['r2'] for tid in task_ids]
        mse_scores = [self.task_metrics[tid]['mse'] for tid in task_ids]
        mae_scores = [self.task_metrics[tid]['mae'] for tid in task_ids]

        # Set up plot - 修复这一行
        fig = plt.figure(figsize=(12, 7))

        # Create bar positions
        x = np.arange(len(task_ids))
        width = 0.25

        # Plot bars - 修复这一行
        ax1 = fig.add_subplot(111)
        bars1 = ax1.bar(x - width, r2_scores, width, label='R²', color='royalblue')

        # Create second y-axis for MSE and MAE
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x, mse_scores, width, label='MSE', color='tomato')
        bars3 = ax2.bar(x + width, mae_scores, width, label='MAE', color='green')

        # Add horizontal line for good performance threshold (R²=0.7)
        ax1.axhline(y=0.7, color='blue', linestyle='--', alpha=0.5, label='Good Performance Threshold (R²=0.7)')

        # Set labels and title
        ax1.set_xlabel('Tasks')
        ax1.set_ylabel('R² Score')
        ax2.set_ylabel('MSE/MAE Score')
        plt.title('Final Performance Across Tasks')

        # Set task labels
        ax1.set_xticks(x)
        ax1.set_xticklabels(task_ids, rotation=45)

        # Add dimension info to labels
        labels = [f"{tid}\n({self.task_dims.get(self.task_id_mapping.get(tid, tid), '?')} features)"
                  for tid in task_ids]
        ax1.set_xticklabels(labels, rotation=45, ha='right')

        # Add values on top of bars
        def add_labels(bars, ax):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        add_labels(bars1, ax1)
        add_labels(bars2, ax2)
        add_labels(bars3, ax2)

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)

        # Adjust layout and save
        plt.tight_layout()
        filename = os.path.join(self.visualization_dir, 'performance_comparison.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance comparison saved to {filename}")

    def analyze_task_relationships(self):
        """Analyze and visualize relationships between tasks"""
        if not self.task_similarities or len(self.task_metrics) < 2:
            print("Not enough task data for relationship analysis")
            return

        # Create a network graph of task relationships
        try:
            import networkx as nx

            # Create graph
            G = nx.Graph()

            # Add nodes
            for task_id in self.task_metrics:
                # Get dimension info for node label
                internal_tid = self.task_id_mapping.get(task_id, task_id)
                dim = self.task_dims.get(internal_tid, '?')
                G.add_node(task_id, dim=dim)

            # Add edges for similar tasks
            for task1, similarities in self.task_similarities.items():
                for task2, sim_value in similarities.items():
                    if sim_value > self.similarity_threshold:
                        G.add_edge(task1, task2, weight=sim_value)

            # Plot
            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(G, seed=42)

            # Draw nodes with size based on dimension
            node_sizes = [G.nodes[n].get('dim', 1) * 20 + 300 for n in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue')

            # Draw edges with width based on similarity
            edge_weights = [G.get_edge_data(u, v)['weight'] * 5 for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7)

            # Draw node labels
            node_labels = {n: f"{n}\n({G.nodes[n].get('dim', '?')}D)" for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

            # Draw edge labels
            edge_labels = {(u, v): f"{G.get_edge_data(u, v)['weight']:.2f}" for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

            plt.title("Task Relationship Network (Edge Width = Similarity Strength)")
            plt.axis('off')
            plt.tight_layout()

            # Save figure
            filename = os.path.join(self.visualization_dir, 'task_relationships.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Task relationships network saved to {filename}")

        except ImportError:
            print("NetworkX library not available for relationship visualization")

    def analyze_parameter_transfer(self, task_id1, task_id2):
        """Compare task-specific HEAD parameters between two tasks.

        With the new architecture there is a single, globally-shared
        backbone, so a "shared-layer cosine similarity between tasks"
        no longer makes sense (both tasks now read from the very same
        weights). We instead compare the per-task HEAD parameters,
        which still differ across tasks.
        """
        internal_task_id1 = self.task_id_mapping.get(task_id1, task_id1)
        internal_task_id2 = self.task_id_mapping.get(task_id2, task_id2)

        if internal_task_id1 not in self.task_networks or internal_task_id2 not in self.task_networks:
            print(f"Both tasks must be trained before analyzing parameter transfer")
            return

        head1 = self.task_networks[internal_task_id1]
        head2 = self.task_networks[internal_task_id2]

        params1 = head1.get_task_params()
        params2 = head2.get_task_params()

        param_similarities = {}
        for name in params1:
            if name in params2:
                p1 = params1[name].view(-1).detach().cpu().numpy()
                p2 = params2[name].view(-1).detach().cpu().numpy()
                if p1.shape == p2.shape and np.linalg.norm(p1) > 0 and np.linalg.norm(p2) > 0:
                    sim = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
                    param_similarities[name] = sim

        if param_similarities:
            plt.figure(figsize=(10, 6))
            names = list(param_similarities.keys())
            sims = [param_similarities[n] for n in names]

            plt.bar(range(len(names)), sims)
            plt.xlabel('Task-head parameter')
            plt.ylabel('Cosine similarity')
            plt.title(f'Task-head parameter similarity between {task_id1} and {task_id2}')
            plt.xticks(range(len(names)), [n.split('.')[-1] for n in names], rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()

            filename = os.path.join(self.visualization_dir, f'parameter_transfer_{task_id1}_{task_id2}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Parameter transfer analysis saved to {filename}")

            # Return average similarity
            avg_sim = np.mean(list(param_similarities.values()))
            print(f"Average parameter similarity: {avg_sim:.4f}")
            return avg_sim
        else:
            print("No comparable parameters found between tasks")
            return None

    def save_results(self):
        """Save model results and metrics to file"""
        # Create results dictionary
        results = {
            'task_metrics': self.task_metrics,
            'task_similarities': self.task_similarities,
            'task_dims': self.task_dims,
        }

        if hasattr(self, 'forgetting_metrics'):
            results['forgetting_metrics'] = self.forgetting_metrics

        # Save as pickle file
        import pickle
        os.makedirs(self.visualization_dir, exist_ok=True)
        filename = os.path.join(self.visualization_dir, 'model_results.pkl')

        with open(filename, 'wb') as f:
            pickle.dump(results, f)

        print(f"Model results saved to {filename}")