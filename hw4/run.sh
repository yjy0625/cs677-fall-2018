: 'run.sh'
IMGDIR='./HW4_data'

python3 hw4.py --imgdir=$IMGDIR --img1=a1.png --img2=a2.png --outdir=result_a --latex
python3 hw4.py --imgdir=$IMGDIR --img1=b1.png --img2=b2.png --outdir=result_b --latex
python3 hw4.py --imgdir=$IMGDIR --img1=c1.png --img2=c2.png --outdir=result_c --latex
