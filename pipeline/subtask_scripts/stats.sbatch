#!/usr/bin/bash

echo $LABELMAKER_REPO
echo $TARGET_DIR

export SACCT_FORMAT="JobID,JobName,User,Partition,NodeList,Elapsed,State,ExitCode,MaxRSS,AveRSS,AllocTRES,AveCPU,AveCPUFreq,CPUTime" 

for jobid in $(echo ${monitor_jobids[@]})
do 
  (sacct -j $jobid && seff $jobid) > $log_dir/${video_id}/stats_${jobid}.out
done 

chgrp -R INFK-CVG-Euler-All ${target_dir}
