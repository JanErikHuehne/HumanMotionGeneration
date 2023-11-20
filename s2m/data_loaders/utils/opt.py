from argparse import Namespace
import re
from os.path import join as pjoin


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag

def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    
    if str(numStr).isdigit():
        flag = True
    return flag

def get_opt(opt_path, device):

    opt = Namespace()
    opt_dict = vars(opt)
    print('Reading', opt_path)
    skip = ('-------------- End ----------------',
           '------------ Options -------------',
           '\n')
    with open(opt_path) as f:
        for line in f:
            if line.strip() not in skip:
                    key, value = line.strip().split(": ")
                    if value in ('True', 'False'):
                        opt_dict[key] = bool(value)
                    elif is_float(value):
                        opt_dict[key] = float(value)
                    elif is_number(value):
                        opt_dict[key] = int(value)
                    else:
                        opt_dict[key] = str(value)
              
    
    opt.joint_num = 22
    opt.dim_pose = 263
    opt.max_motion_length = 196
    opt.is_train = True
    opt.device = device
    opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    return opt