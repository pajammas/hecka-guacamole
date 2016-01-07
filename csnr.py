# Translated to Python by Duncan Sommer

from numpy import *

def csnr(A,B,row,col):
   
   if len(B.shape) == 2:
      [n,m] = B.shape
      ch = 1
   else:
      [n,m,ch] = B.shape

   #A=double(A)
   #B=double(B)
   
   e = A-B
   e = e[row+1 : n-row][col+1 : m-col]

   if ch == 1:
      me = mean(mean(square(e)))      
   
   else:
      me1 = mean(mean(square(e[:,:,0])))
      me2 = mean(mean(square(e[:,:,1])))
      me3 = mean(mean(square(e[:,:,2])))
      me = mean(me1, me2, me3)      
   
   s = 10*log10(255*255/me)

   return s