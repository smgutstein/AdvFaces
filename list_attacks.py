import importlib
import inspect

def get_attacks():
    attack_module = importlib.import_module("foolbox.foolbox.attacks")
    attack_mod_classes = [xx
                          for xx in inspect.getmembers(attack_module)
                          if inspect.isclass(xx[1])]
    temp_dict = {xx[0]:xx[1] for xx in attack_mod_classes}
    attack_dict = {xx:temp_dict[xx]
                   for xx in temp_dict
                   if (issubclass(temp_dict[xx], temp_dict['Attack']) and xx != 'Attack')}

    return attack_dict

if __name__ == '__main__':
    with open("attacks.txt",'w') as f:
        ad = get_attacks()
        for curr_attack in sorted(ad):
            f.write(curr_attack + '\n')
