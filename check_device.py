import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

checkpoint = torch.jit.load("checkpoints\Wav2Lip-SD-GAN.pt", map_location="cpu")
s = checkpoint.state_dict()
new_s = {}
for k, v in s.items():
		new_s[k.replace('module.', '')] = v
print(new_s.keys())