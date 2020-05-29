# 递归搜索文件
function ClangFormatFile()
{
  # for file in `ls $1`　　　　　　
   for file in $(ls $1)　
   do
    #  echo $1"/"$file
     if [ -d $1'/'$file ]; then
       ClangFormatFile $1'/'$file
     else
      #  echo $1'/'$file
       if [ "${file##*.}"x = "h"x ] ||
       [ "${file##*.}"x = "cc"x ] ||
       [ "${file##*.}"x = "m"x ] || [ "${file##*.}"x = "mm"x ]; then
         clang-format -i $1'/'$file
       fi
     fi
   done
}

echo '------clang-format start: '$1
ClangFormatFile $1
echo '------clang-format end'
