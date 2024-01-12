import os, shutil, glob, re, torch
torch_name  = 'torch' + (torch.__version__.split('+')[0] + 'cu' + torch.version.cuda).replace('.', '') 
# get the .whl file in the dist directory and rename 
file = glob.glob(os.path.join('dist', '*.whl'))
current_dir = os.getcwd()
if file:
    file = file[0]
    insert_pos = re.search(r'-cp[0-9]+-cp[0-9]+-', file).start() 
    new_name = f'{file[:insert_pos]}+{torch_name}{file[insert_pos:]}' 
    print(os.path.join(os.getcwd(), new_name))
    shutil.move(file, new_name)