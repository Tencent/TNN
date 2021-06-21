# 通过./xxx.sh的方式执行脚本
# 即若脚本中未指定解释器，则使用系统默认的shell

# 您可以通过setEnv函数设置插件间传递的参数
# setEnv "FILENAME" "package.zip"
# 然后在后续的插件的表单中使用${FILENAME}引用这个变量

# 您可以在质量红线中创建自定义指标，然后通过setGateValue函数设置指标值
# setGateValue "CodeCoverage" $myValue
# 然后在质量红线选择相应指标和阈值。若不满足，流水线在执行时将会被卡住

# cd ${WORKSPACE} 可进入当前工作空间目录

cd ${WORKSPACE}

pwd
ls -l

# 1. 解压3DMM流水线制品
rm -rf release/
rm -rf result/
unzip release_android.zip
unzip result.zip

# 2. 拷贝iOS库，拷贝资源bundle
cd result
chmod +x copy_file_3dmm.sh
./copy_file_3dmm.sh ${WORKSPACE}"/GYAILib"
cd ..

# 3. 拷贝Android .a和.so
cd release
chmod +x copy_file_3dmm_and.sh
./copy_file_3dmm_and.sh ${WORKSPACE}"/GYAILib"
cd ..

# 4. 推送GIT
GIT_MSG="[pipeline] name = "${BK_CI_PIPELINE_NAME}", buildnum = "${BK_CI_BUILD_NUM}  #流水线名称不能含有 ( )
echo $GIT_MSG

# 4.1. 配置提交作者信息
git config --global user.email "atilazhang@tencent.com"
git config --global user.name "atilazhang"

# 4.2. 推送GYAIThirdPartyLib和GYAILib
cd GYAILib
cd GYAIThirdPartyLib
git fetch  #
echo ${CV_BRANCH}
git checkout ${CV_BRANCH}  #
git pull
git add .
git commit -m  "$GIT_MSG"
git push
cd ..
git pull
git add .
git reset .gitmodules  #.gitmodules会被蓝盾修改，需要忽略
git commit -m  "$GIT_MSG"
git push
cd ..
