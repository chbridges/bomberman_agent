cat coinllector.log | grep "ROUND"  | tail -1 | awk '{print $NF}'
