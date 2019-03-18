while true
do
echo
date
cat cratemate.log | grep "ROUND"  | tail -1 | awk '{print $NF}'
sleep 60
done
