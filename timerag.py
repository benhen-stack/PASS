import numpy as np
from sklearn.cluster import KMeans

class TimeRAG:
    def __init__(self, model, lookback=8, horizon=1, top_k=3, n_clusters=None):
        """
        Zero-Shot TimeRAG Implementation based on arXiv:2412.16643v1 logic.

        Differences from Paper:
        - Uses Text Prototypes instead of Trainable Reprogramming Layer (Section II-C).
        - Zero-shot inference instead of 50-epoch training (Section III-A).
        + Matches DTW Squared Euclidean metric (Eq 2).
        + Matches K-Means Knowledge Base construction (Eq 1).
        """
        self.model = model
        self.L = lookback
        self.H = horizon
        self.K = top_k
        self.n_clusters = n_clusters

        # Knowledge Base Storage
        self.kb_patterns = None  # X_L in paper
        self.kb_targets = None   # Future H steps for the cached patterns

    def _dtw_squared_euclidean(self, s1, s2):
        """
        Calculates DTW distance using Squared Euclidean metric as defined in Eq. (2) .
        d(i,j) = (x_i - y_j)^2
        """
        n, m = len(s1), len(s2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Eq 2: Squared Euclidean Distance
                cost = (s1[i-1] - s2[j-1]) ** 2

                # Dynamic Programming path
                last_min = min(dtw_matrix[i-1, j],    # Insertion
                               dtw_matrix[i, j-1],    # Deletion
                               dtw_matrix[i-1, j-1])  # Match

                dtw_matrix[i, j] = cost + last_min

        return dtw_matrix[n, m]

    def _modality_alignment(self, sequence):
        """
        Aligns sequence modality to text prototypes (Fig 1).
        Approximates the paper's 'Reprogramming Layer' via prompt engineering.
        """
        seq_str = ", ".join([f"{x:.2f}" for x in sequence])
        return f"Sequence values: {seq_str}"

    def fit(self, train_data):
        """
        Constructs Time-Series Knowledge Base (Section II-B).
        """
        n = len(train_data)
        X, y = [], []

        # 1. Sequential Slicing (Sliding Window)
        if n < self.L + self.H:
            raise ValueError("Training data too short.")

        for i in range(n - self.L - self.H + 1):
            window = train_data[i : i + self.L]
            target = train_data[i + self.L : i + self.L + self.H]
            X.append(window)
            y.append(target)

        X = np.array(X)
        y = np.array(y)

        # Dynamic cluster sizing
        if self.n_clusters is None:
            # Heuristic: limit clusters to keep retrieval fast
            self.n_clusters = max(2, min(int(len(X) * 0.1), 100))

        print(f"[TimeRAG] Clustering {len(X)} segments into {self.n_clusters} representative patterns...")

        # 2. K-Means Clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)

        # 3. Knowledge Base Construction: Select closest to centroid
        kb_X_list = []
        kb_y_list = []

        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        for i in range(self.n_clusters):
            indices = np.where(labels == i)[0]
            if len(indices) == 0: continue

            cluster_samples = X[indices]
            center = centers[i]

            # Euclidean distance for selection (standard K-Means metric)
            dists = np.linalg.norm(cluster_samples - center, axis=1)
            best_idx_local = np.argmin(dists)
            best_idx_global = indices[best_idx_local]

            kb_X_list.append(X[best_idx_global])
            kb_y_list.append(y[best_idx_global])

        self.kb_patterns = np.array(kb_X_list)
        self.kb_targets = np.array(kb_y_list)
        print(f"[TimeRAG] KB constructed with {len(self.kb_patterns)} entries.")

    def retrieve(self, query_seq):
        """
        Retrieval Stage: DTW-based similarity search (Section II-C).
        """
        distances = []

        for i, pattern in enumerate(self.kb_patterns):
            dist = self._dtw_squared_euclidean(query_seq, pattern)
            distances.append((dist, i))

        # Select Top K
        distances.sort(key=lambda x: x[0])
        top_indices = [idx for _, idx in distances[:self.K]]

        return [(self.kb_patterns[idx], self.kb_targets[idx]) for idx in top_indices]

    def predict(self, history):
        """
        Forecast Stage (Fig 1).
        Combines retrieved context into a prompt.
        """
        input_seq = history[-self.L:]

        # 1. Retrieve
        retrieved_items = self.retrieve(input_seq)

        # 2. Prompt Construction (Fig 1 Structure)
        reference_text = ""
        for i, (patt, targ) in enumerate(retrieved_items):
            patt_text = self._modality_alignment(patt)
            targ_str = ", ".join([f"{x:.2f}" for x in targ])
            reference_text += f"Reference {i+1}:\n{patt_text}\nNext Value: {targ_str}\n\n"

        input_text = self._modality_alignment(input_seq)

        prompt = f"""
        [Begin Data]
        Input Sequence:
        {input_text}

        [Instruction]
        Predict the next {self.H} steps given the previous {self.L} steps information attached.

        [Reference Data]
        TOP {self.K} Sequences:
        {reference_text}

        Return ONLY the predicted number.
        """
        # 3. LLM Prediction
        try:
            response = self.model.generate_content(prompt)
            import re
            text = response.text
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)

            if len(matches) > 0:
                return [float(matches[0])]
            return [float(input_seq[-1])]

        except Exception as e:
            return [float(input_seq[-1])]