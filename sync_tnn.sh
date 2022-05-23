git remote | grep github-tnn > /dev/null
if test $? != 0; then
    git remote add github-tnn https://github.com/Tencent/TNN.git
fi
git pull origin master
git pull github-tnn master
