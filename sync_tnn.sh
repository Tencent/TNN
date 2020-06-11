git remote | grep tnn > /dev/null
if test $? != 0; then
    git remote add tnn http://git.code.oa.com/deep_learning_framework/TNN.git
fi
git pull origin master
git pull tnn master
