while true
do
echo
date
printf "coinllector:   "
cat ./coinllector/logs/coinllector.log | grep "ROUND"  | tail -1 | awk '{print $NF}'
sleep 60
done
