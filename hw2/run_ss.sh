: 'run_ss.sh'
IMGDIR='./HW2_ImageData/Images'
GTDIR='./HW2_ImageData/boundingbox_groudntruths'
OUTDIR='./out_ss'

python3 selective_search.py --max_rects=100 --strategy=color --imgdir=$IMGDIR --gtdir=$GTDIR --outdir=$OUTDIR
python3 selective_search.py --max_rects=100 --strategy=multi --imgdir=$IMGDIR --gtdir=$GTDIR --outdir=$OUTDIR