#！/bin/bash
git add .  &&
if [ ! -n "$1" ] ;then
    m="update"
else
    m="$1"
fi
echo "更新信息为:$m"
git commit -m"$m"  &&
git push &&
echo "git更新成功！"
