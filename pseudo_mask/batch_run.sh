
cwd=$(pwd)
for img in $cwd/train/*;
    do
        echo $img
        python main.py
    done
