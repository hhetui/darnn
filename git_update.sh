#！/bin/bash
git add .  &&
if [ ! -n "$1" ] ;then
    m="update"
else
    m="$1"
fi
git commit -m"$m"  &&
git push

