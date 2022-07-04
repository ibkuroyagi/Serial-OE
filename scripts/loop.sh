#!/bin/bash
rm -rf dump/dev dump/eval
ln -s /fsws1/i_kuroyanagi/dcase2020/dump/dev dump
ln -s /fsws1/i_kuroyanagi/dcase2020/dump/eval dump
echo "Finish Copy!"
