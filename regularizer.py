import torch
import torch.nn.functional as F


# class DifferentialEntropyRegularization(torch.nn.Module):

#     def __init__(self, eps=1e-8):
#         super(DifferentialEntropyRegularization, self).__init__()
#         self.eps = eps
#         self.pdist = torch.nn.PairwiseDistance(2)

#     def forward(self, x):

#         with torch.no_grad():
#             dots = torch.mm(x, x.t())
#             n = x.shape[0]
#             dots.view(-1)[::(n + 1)].fill_(-1)  # trick to fill diagonal with -1
#             _, I = torch.max(dots, 1)  # max inner prod -> min distance

#         rho = self.pdist(x, x[I])

#         # dist_matrix = torch.norm(x.unsqueeze(1) - x.unsqueeze(0), p=2, dim=-1)
#         # rho = dist_matrix.topk(k=2, largest=False)[0][:, 1]

#         loss = -torch.log(rho + self.eps).mean()

#         return loss


#knn
class DifferentialEntropyRegularization(torch.nn.Module):
    def __init__(self, k=5, eps=1e-8):
        super(DifferentialEntropyRegularization, self).__init__()
        self.k = k
        self.eps = eps
        self.pdist = torch.nn.PairwiseDistance(2)

    def forward(self, x):
        # x: (batch_size, embed_dim)
        with torch.no_grad():
            # Normalize x to unit sphere
            x = torch.nn.functional.normalize(x, dim=1)

            # Compute cosine similarity matrix
            dots = torch.mm(x, x.t())  # (N, N)

            n = x.shape[0]
            dots.view(-1)[::(n + 1)].fill_(-1)  # Fill diagonal with -1 to ignore self-similarity

            # Find top-k most similar vectors for each sample
            topk_values, topk_indices = torch.topk(dots, self.k, dim=1, largest=True)

        # Gather the corresponding neighbor embeddings
        x_neighbors = x[topk_indices]  # (batch_size, k, embed_dim)

        # Expand x to (batch_size, k, embed_dim) to match x_neighbors shape
        x_expanded = x.unsqueeze(1).expand_as(x_neighbors)

        # Compute Euclidean distances between x and its top-k neighbors
        distances = torch.norm(x_expanded - x_neighbors, p=2, dim=-1)  # (batch_size, k)

        # Mean distance to k neighbors
        mean_rho = distances.mean(dim=1)  # (batch_size,)

        # KoLeo-kNN loss: maximize log(mean distance)
        loss = -torch.log(mean_rho + self.eps).mean()

        return loss


# thewem nhãn +knn
# class DifferentialEntropyRegularization(torch.nn.Module):
#     def __init__(self, k=5, eps=1e-8):
#         super(DifferentialEntropyRegularization, self).__init__()
#         self.k = k
#         self.eps = eps

#     def forward(self, x, labels):
#         """
#         x: Tensor of shape (N, D) - embedding vectors
#         labels: Tensor of shape (N,) - class labels
#         """
#         with torch.no_grad():
#             x = F.normalize(x, dim=1)

#             # Cosine similarity matrix
#             sim_matrix = torch.matmul(x, x.T)  # (N, N)

#             n = x.shape[0]
#             sim_matrix.view(-1)[::n + 1] = -1  # exclude self-similarity (diagonal)

#             # Mask to keep only same-class pairs
#             label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
#             sim_matrix[~label_mask] = -1  # remove different-class similarities

#             # Get top-k same-class neighbors for each point
#             topk_values, _ = torch.topk(sim_matrix, self.k, dim=1, largest=True)

#         # Convert similarity to distance (cosine distance)
#         distances = 1 - topk_values  # (N, k)

#         # Normalize distances into probabilities
#         distances = distances / (distances.sum(dim=1, keepdim=True) + self.eps)

#         # Entropy per sample
#         entropy = - (distances * torch.log(distances + self.eps)).sum(dim=1)  # (N,)

#         # Final loss: maximize entropy → minimize negative entropy
#         loss = -entropy.mean()

#         return loss
    
    
# thêm nhãn và ko dùng knn

# class DifferentialEntropyRegularization(torch.nn.Module):
#     def __init__(self, eps=1e-8):
#         super(DifferentialEntropyRegularization, self).__init__()
#         self.eps = eps

#     def forward(self, x, labels):
#         """
#         x: Tensor of shape (N, D) - embedding vectors
#         labels: Tensor of shape (N,) - class labels
#         """
#         with torch.no_grad():
#             x = F.normalize(x, dim=1)

#             # Cosine similarity matrix
#             sim_matrix = torch.matmul(x, x.T)  # (N, N)
#             n = x.shape[0]

#             # Bỏ tự so sánh bản thân (diagonal)
#             sim_matrix.view(-1)[::n + 1] = -1

#             # Tạo mask cùng lớp
#             label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
#             sim_matrix = sim_matrix.masked_fill(~label_mask, -1)  # Loại bỏ khác lớp

#             # Loại bỏ -1 (khác lớp hoặc chính mình), chỉ giữ điểm hợp lệ
#             distances = 1 - sim_matrix  # (N, N), cosine distance
#             valid_mask = sim_matrix > -1  # chỉ giữ các điểm cùng lớp và khác bản thân

#             # Chuẩn hóa thành phân phối xác suất
#             distances = distances * valid_mask  # zero-out những điểm không hợp lệ
#             row_sums = distances.sum(dim=1, keepdim=True) + self.eps
#             probs = distances / row_sums  # (N, N)

#             # Tính entropy theo hàng
#             entropy = - (probs * torch.log(probs + self.eps)).sum(dim=1)  # (N,)

#             # Final loss
#             loss = -entropy.mean()

#         return loss


# nograD
# class DifferentialEntropyRegularization(torch.nn.Module):
#     def __init__(self, eps=1e-8):
#         super(DifferentialEntropyRegularization, self).__init__()
#         self.eps = eps

#     def forward(self, x, labels):
#         """
#         x: Tensor of shape (N, D) - embedding vectors
#         labels: Tensor of shape (N,) - class labels
#         """
#         x = F.normalize(x, dim=1)  # vẫn cần normalize để cosine similarity đúng

#         # Tính cosine similarity
#         sim_matrix = torch.matmul(x, x.T)  # (N, N)
#         n = x.shape[0]
#         sim_matrix.view(-1)[::n + 1] = -1  # loại bỏ self-similarity (diagonal)

#         # Tạo mask cùng class (không cần gradient)
#         with torch.no_grad():
#             label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)

#         # Áp dụng mask để loại bỏ khác class
#         sim_matrix = sim_matrix.masked_fill(~label_mask, -1)

#         # Chuyển sang cosine distance
#         distances = 1 - sim_matrix  # (N, N)
#         valid_mask = sim_matrix > -1  # chỉ giữ điểm hợp lệ (cùng class)

#         distances = distances * valid_mask  # zero-out các điểm không hợp lệ
#         row_sums = distances.sum(dim=1, keepdim=True) + self.eps
#         probs = distances / row_sums  # (N, N)

#         # Tính entropy
#         entropy = - (probs * torch.log(probs + self.eps)).sum(dim=1)  # (N,)
#         loss = -entropy.mean()  # maximize entropy → minimize negative entropy

#         return loss