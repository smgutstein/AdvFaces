with open("attacks.txt","r") as f, open("all_tests.sh",'w') as ad:
    for curr_attack in f.readlines():
        ad.write("python foolbox_test5a.py -a " + curr_attack)
