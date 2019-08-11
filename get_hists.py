import os

attack_list = []
with open("success.txt","r") as f:
    data = f.readlines()
    curr_attack = None
    curr_acc = None
    done = False

    for curr_line in data:
        if "OK" in curr_line:
            attack_name = curr_line.split(":")[0]
            print(attack_name, end = ' ')
        elif "Raw Image" in curr_line:
            true_cat = curr_line.split(" -- ")[-2].split(',')[0].strip()
            true_conf = curr_line.split(" -- ")[-1].strip()
            print (true_cat, "  ", true_conf , end=' ')            
        elif "Adv Image" in curr_line:
            adv_cat = curr_line.split(" -- ")[-2].split(',')[0].strip()
            adv_conf = float(curr_line.split(" -- ")[-1].strip())
            print (adv_cat, "  ", adv_conf)            
            done = True

        if done:
            data_dir = os.path.join(true_cat, attack_name)
            print(os.path.join(data_dir, "hist.png"))
            attack_list.append((attack_name, true_cat, adv_cat,
                                adv_conf, data_dir))
            done = False

    attack_list = sorted(attack_list, key=lambda x:x[3], reverse=True)
