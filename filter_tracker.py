with open("tracker_indivs.txt","r") as f:
    
    borders = "-------------"
    complete_attack_results = False
    num_borders = 0
    out_str = ""
    ok_ctr = 0
    bad_ctr = 1
    with open("success.txt","w") as f1, open("failed.txt","w") as f2:
        for curr_line in f.readlines():
            out_str += curr_line
            if borders in curr_line:
                num_borders += 1
            if num_borders == 2:
                if "FAILED" in out_str:
                    f2.write(out_str)
                    bad_ctr += 1
                else:
                    f1.write(out_str)
                    ok_ctr += 1
                out_str = curr_line
                num_borders = 1

        ok_str =  borders + '\n\n' + str(ok_ctr) + ' ' + 'Successes\n'
        bad_str = borders + '\n\n' + str(bad_ctr) + ' ' + 'Failures\n'
        f1.write(ok_str)
        f2.write(bad_str)

