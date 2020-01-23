BEGIN{
    for(i=0;i<3;i++){
	arr[i] = i*2+1;
	count[i] = 0;
    }
    total = 0;
}
{
    if ($0 ~/^Images/){
       filename = $2;
       split($2,a,".");
       query_str = a[1]; 
       printf(" new image: %s\n",query_str);
       topcount = 1;
       total++;
    }
    else {
        if (query_str == $1){
	    printf("MAtch top count is %d\n",topcount);
	    for(j=0;j<3; j++) {
	        if (topcount <= arr[j]){
		    count[j] = count[j] + 1;
	        }
	    }
        }
	topcount++;
    }
}
END{
    printf("Total Image: %d\n", total);
    for(i=0;i<3;i++){
        printf("%d  %d %d\n",i,arr[i],count[i]);
    }
    for(i=0;i<3;i++){
        printf("%d  %d %3.3f\n",i,arr[i],count[i]/total*100);
    }
}
