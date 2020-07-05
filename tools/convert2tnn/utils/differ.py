# Tencent is pleased to support the open source community by making TNN available.
#
# Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
import os
import sys
import math

error_up_limits = 0.001

def is_number(s):
    try:
        if math.isinf(float(s)) or math.isnan(float(s)):
            return False
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        value = unicodedata.numeric(s)
        if math.isinf(value) or math.isnan(value):
            return False
        return True
    except (TypeError, ValueError):
        pass
    return False


def remove_invalid_number(lines):
    valid_lines=[]
    for i in range(0, len(lines)):
        if is_number(lines[i]):
            valid_lines.append(lines[i])
        else:
            print("there are some error number: " + str(lines[i]))
            #exit(-1)
    return valid_lines

def compare_files(file_name1, file_name2):

    if file_name1.endswith('\n'):
        file_name1 = file_name1[:-1]

    if file_name2.endswith('\n'):
        file_name2 = file_name2[:-1]

    if not os.path.isfile(file_name1):
        print ('error:file:%s. doesn\'t exist.' % file_name1)
        return -1

    if not os.path.isfile(file_name2):
        print ('error:file:%s. doesn\'t exist.' % file_name2)
        return -1

    print ('comparing %s and %s' % (file_name1, file_name2))

    max_diff = 0.0
    max_diff_line_number = -1
    with open(file_name1) as file1:
        with open(file_name2) as file2:
            lines1 = file1.readlines()
            lines2 = file2.readlines()

            #strip invalid lines
            valid_lines1 = remove_invalid_number(lines1)
            valid_lines2 = remove_invalid_number(lines2)

        if len(valid_lines1) != len(valid_lines2):
            print ('the number of lines is not equal in file %s and %s' % (file_name1, file_name2))
            print ( str(len(valid_lines1))+ " vs " + str(len(valid_lines2)))
            return -1

        # file_name1 and file_name2 have same number of lines, let's compare these two files
        for i in range (0, len(lines1)):
            #print 'processing line %d of %d in file %s and %s' % (i+1, len(lines1), file_name1, file_name2)
                is_line1_valid = is_number(lines1[i])
                is_line2_valid = is_number(lines2[i])
                if not is_line1_valid and not is_line2_valid:
                    continue
                if not is_line1_valid:
                   print ("error: line %d in file %s is a NOT valid number while %d in file %s is a valid number" %
                           (i+1, file_name1, i+1, file_name2))
                   return -1
                if not is_line2_valid:
                   print ("error: line %d in file %s is a valid number while %d in file %s is NOT a valid number" %
                           (i+1, file_name1, i+1, file_name2))
                   return -1

                number1 = float(lines1[i])
                number2 = float(lines2[i])
                diff = number1 - number2
                if diff < 0:
                    diff = -1*diff

                if diff >= error_up_limits:
                    print ('%s and %s differ %f at line %d with %f and %f' % (file_name1, file_name2, diff, i+1,
                        number1, number2))
                    return -1

                if diff > max_diff:
                    max_diff = diff
                    max_diff_line_number = i+1
    if(max_diff_line_number == -1):
        print ('max_diff is 0.')
    else:
        print ('max_diff is %f at line %d' % (max_diff, max_diff_line_number))

    return 0

if len(sys.argv) != 3 and len(sys.argv) != 4:
    print ('usage(1):python %s layer_output_files_list_file1 layer_output_files_list_file2' % sys.argv[0])
    print ('usage(2):python %s layer_output_files_list_file1 layer_output_files_list_file2 starting_file_id(zero-based)'
            % sys.argv[0])
    exit(-1)

file_name1 = sys.argv[1];
file_name2 = sys.argv[2];
if compare_files(file_name1, file_name2) != 0:
    exit(-1)

print ('all the numbers in the file %s and %s are aligned.' % (file_name1, file_name2))
print ("the differ limits:" + str(error_up_limits))
