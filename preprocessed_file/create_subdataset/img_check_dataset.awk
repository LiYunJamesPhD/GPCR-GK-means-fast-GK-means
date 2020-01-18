awk '

    FNR==NR{
        # process file file_1 (r18)
        if ($0 ~ /.JPEG/){
	    split($0, tmp, " ");
            label_line = tmp[2];
            split(label_line, array, ".");

	    # get the next line
	    getline;
	    split($0, tmp2, " ");
	    #printf("result : %s", array[1]);
	    if (array[1] == tmp2[1] && tmp2[2] > 0.795){
	        arry[FNR] = label_line;
                #printf("result %s\n", label_line);
		#printf("result 2 %s\n", $0);
	    }
	    #arry[FNR] = $0;
	}
        next 
    } 
    {
        # process file file_2 (iv3)
        if ($0 ~ /.JPEG/){
	    split($0, tmp_b, " ");
            label_line_b = tmp_b[2];
            split(label_line_b, array_b, ".");

	    # get the next line
            getline;
	    split($0, tmp2_b, " ");
	    if (array_b[1] == tmp2_b[1] && tmp2_b[2] > 0.795){
		# check whether or not the label is in arry.
	        for (key in arry){
		    if (label_line_b == arry[key]){
		        printf("%s\n", label_line_b);
		    }
		}
	    }
	    #print arry[FNR]
	}
        #print arry[FNR], $0 
    }

' ~/Desktop/squared_imgs_5000_result_iv3 ~/Desktop/squared_imgs_5000_result_r50
