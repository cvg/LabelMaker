#!/usr/bin/bash

echo $LABELMAKER_REPO
echo $TARGET_DIR
echo $VIDEO_ID
echo $MONITOR_JOBIDS
echo $LOG_DIR

export SACCT_FORMAT="JobID,JobName,User,Partition,NodeList,Elapsed,State,ExitCode,MaxRSS,AveRSS,AllocTRES,AveCPU,AveCPUFreq,CPUTime"

for jobid in $(echo ${MONITOR_JOBIDS[@]}); do
  (sacct -j $jobid && seff $jobid) >$LOG_DIR/${VIDEO_ID}/stats_${jobid}.out
done

chgrp -R INFK-CVG-Euler-All ${TARGET_DIR}
