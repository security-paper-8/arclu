git clone https://github.com/tjmoon0104/pytorch-tiny-imagenet.git  ./pytorch-tiny-imagenet
cd ./pytorch-tiny-imagenet
bash run.sh
cd tiny-224
mv train train_unselected
mv "test" test_unselected
mkdir train
mkdir "test"

for CID in  n01774750 n01855672 n02056570 n02165456 n02279972 n02504458 n02815834 n02917067 n03026506 n03085013 n03100240 n03662601 n04099969 n04146614 n04285008 n04486054 n07720875 n07734744 n07873807 n07920052
do
    cp -r train_unselected/${CID} ./train/
    cp -r test_unselected/${CID} ./test/
done

