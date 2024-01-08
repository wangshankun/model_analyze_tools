import numpy as np

def cosine_similarity(a, b):
    cos_sim = b.dot(a) / (np.linalg.norm(b) * np.linalg.norm(a))
    return cos_sim

def compute_cosine_similarity_for_slices(tensor_A, tensor_B, axis=1):
    """
    计算两个张量在指定轴上各个切片的余弦相似度。
    tensor_A, tensor_B: 形状相同的两个张量
    axis: 要操作的轴，这里默认为1
    """
    # 验证张量形状相同
    if tensor_A.shape != tensor_B.shape:
        raise ValueError("两个张量的形状必须相同")

    # 获取指定轴的大小
    dim = tensor_A.shape[axis]

    # 用于存储相似度的数组
    similarity_values = np.zeros(dim)

    # 遍历指定轴
    for i in range(dim):
        # 提取切片
        slice_A = tensor_A.take(i, axis=axis)
        slice_B = tensor_B.take(i, axis=axis)

        # 计算余弦相似度并存储
        similarity_values[i] = cosine_similarity(slice_A.flatten(), slice_B.flatten())

    return similarity_values

# 假设tensor_A和tensor_B是两个形状为(1, 64, 54, 54)的张量
# 这里使用随机数据作为示例
np.random.seed(0)
tensor_A = np.random.rand(1, 64, 54, 54)
tensor_B = np.random.rand(1, 64, 54, 54)

tensor_A = np.load("/root/work/mqbench/application/imagenet_example/PTQ/ptq/1002.npy")
tensor_B = np.load("/root/work/mqbench/application/imagenet_example/PTQ/ptq/torch_layer_dump_1226/maxpool_post_act_fake_quantizer.npy")
tensor_B = tensor_B / 0.47546079754829407
# 计算余弦相似度
similarity_values = compute_cosine_similarity_for_slices(tensor_A, tensor_B)
print(similarity_values)

