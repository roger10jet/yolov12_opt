import torch
# from flash_attn import bert_padding_func, flash_attn_func, flash_attn_qkvpacked_func
#
# x = torch.randn(1, 10, 512, dtype=torch.float16, device='cuda')
# y = torch.randn(1, 10, 512, dtype=torch.float16, device='cuda')
# mask = torch.ones((10,), dtype=torch.bool, device='cuda')
#
# output = flash_attn_func(x, y, mask, causal=False)

A = torch.rand(10, 5)
B = torch.rand(5, 15)
print(B.transpose(1, 0))
