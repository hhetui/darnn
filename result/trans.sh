#！/bin/bash

path="/home/wxk/darnn/result"
ip="ysong@172.17.145.191:/home/ysong/wxk/darnn/result/"
scp -r  $ip$1 $path
