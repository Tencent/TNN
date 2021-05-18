# 递归搜索文件夹下的所有metal文件并编译
# 1: metal文件所在的文件夹
# 2: metallib输出路径


TNNMetalDirectory=$1
TNNMetallibPath=$2
TNNMetalInclude=$3

TNNMetalFloat32=$4

TNNAllMetalFiles=()

function FindAllMetalFiles()
{
  # for file in `ls $1`　　　　　　
   for file in $(ls $1)
   do
     if [ -d $1'/'$file ]; then
       FindAllMetalFiles $1'/'$file
     else
       if [ "${file##*.}"x = "metal"x ]; then
         TNNAllMetalFiles+=($1'/'$file)
       fi
     fi
   done
}

function BuildMetalLib()
{
   TNNAllMetalAIRFiles=()

   #build air files
   for file in ${TNNAllMetalFiles[@]}
   do
      echo "\033[32m       Compile ${file}\033[0m"
      # echo 'TNNMetalFloat32 = '${TNNMetalFloat32}
      xcrun -sdk macosx metal -std=osx-metal1.1 -DTNN_METAL_FULL_PRECISION -dM -I ${TNNMetalInclude} -c ${file} -o ${file}.air
      TNNAllMetalAIRFiles+=(${file}.air)
   done

   #build metallib
   xcrun -sdk macosx metallib ${TNNAllMetalAIRFiles[@]} -o ${TNNMetallibPath}

   #delete air files
   for file in ${TNNAllMetalAIRFiles[@]}
   do
      rm -r ${file}
   done
}

FindAllMetalFiles ${TNNMetalDirectory}
BuildMetalLib


# #Build tnn metallib
# #1: input metal files full path list
# #2: output metallib full path
# function BuildMetal()
# {
#
# }
