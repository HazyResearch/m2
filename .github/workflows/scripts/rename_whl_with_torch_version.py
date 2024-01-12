import os, shutil, glob, re
import torch

torch_name  = 'torch' + (torch.__version__.split('+')[0] + '-cu' + torch.version.cuda).replace('.', '') 
arch_last = os.environ.get('CUDA_ARCH_LIST',';').split(';')[-1].replace('+', '') 
# get the .whl file in the dist directory and rename 
file = glob.glob(os.path.join('dist', '*.whl'))
if file:
    file = file[0]
    insert_pos = re.search(r'cp[0-9]+-cp[0-9]+-', file).start() 
    new_name = f'{file[:insert_pos]}{torch_name}{arch_last}-{file[insert_pos:]}' 
    print(f'move whl from {file} to {new_name}') 
    shutil.move(file, new_name)