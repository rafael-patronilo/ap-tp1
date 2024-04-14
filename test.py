# import os
import itertools

# already_tested = set()
# layers_size = [256,128,64,32]
# layers_size = [x for x in layers_size for _ in range(1, 3)]
# for i in range(1,5):
#     combinations = list(itertools.combinations(layers_size,i))
#     for x in combinations:
#         if x not in already_tested:
#             print(f'Using layers {list(x)}')
#             already_tested.add(x)
        
# def save_last_n(model, name, n):
#     file = f'{name}_{n-1}.pth'
#     if os.path.isfile(file):
#         os.remove(file)
#     for i in reversed(range(1, n)):
#         old_file = f'{name}_{i-1}.pth'
#         file = f'{name}_{i}.pth'
#         if os.path.isfile(old_file):
#             os.rename(old_file, file)
#     with open(f"{name}_0.pth", "w") as f:
#         f.write(model)

# for i in range(10):
#     save_last_n(input(), "model", 3)

def test_vgg_architectures():
    out_channels = [64, 128, 256, 512, 512]
    all_conv_numbers = [x for x in itertools.product(range(1, 3), repeat=len(out_channels)) if sum(x) <= 8]
    all_conv_numbers.extend([x for x in itertools.product(range(1, 3), repeat=len(out_channels)-1) if sum(x) <= 6])
    linear_layer_sizes = [256, 1024, 4096]
    all_linear_layers = zip(linear_layer_sizes, linear_layer_sizes)
    for i, (conv_numbers, linear_layers) in enumerate(itertools.product(all_conv_numbers, all_linear_layers)):
        vgg_blocks = zip(conv_numbers, out_channels)
        print(i, tuple(vgg_blocks), linear_layers)

test_vgg_architectures()