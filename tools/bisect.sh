#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
echo $parent_path
cd "$parent_path"/..
pwd
cd tests
pwd
../../testcode/bin/testcode.py -c continuous
