import itertools

already_tested = set()
layers_size = [256,128,64,32]
layers_size = [x for x in layers_size for _ in range(1, 3)]
for i in range(1,5):
    combinations = list(itertools.combinations(layers_size,i))
    for x in combinations:
        if x not in already_tested:
            print(f'Using layers {list(x)}')
            already_tested.add(x)
        
        