clc;clear;close all

ut = mystats;
x = 1:1:10;


ut.stats(x)
ut.mymean(x,10)
ut.mymedian(x,10)