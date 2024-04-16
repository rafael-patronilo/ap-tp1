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
    out_channels = [16, 32, 64, 128, 256, 512, 512]
    def neuron_count(architecture):
        conv_numbers, linear_layers = architecture
        return sum(linear_layers) + sum(out*count for out, count in zip(out_channels, conv_numbers))
    all_conv_numbers = [x for x in itertools.product(range(4), repeat=len(out_channels)) 
                        if sum(x) >= 1 and sum(1 for y in x if y > 0) >= 2]
    linear_layer_sizes = [32, 64, 128, 256, 1024, 4096]
    all_linear_layers = [x for layer_size in linear_layer_sizes for x in [[layer_size], [layer_size, layer_size]]]
    all_linear_layers.append([])
    architectures = list(itertools.product(all_conv_numbers, all_linear_layers))
    architectures.sort(key=neuron_count)
    for i, architecture in enumerate(architectures[:100]):
        conv_numbers, linear_layers = architecture
        vgg_blocks = zip(conv_numbers, out_channels)
        print(i, neuron_count(architecture), tuple(block for block in vgg_blocks if block[0]>0), linear_layers)

test_vgg_architectures()