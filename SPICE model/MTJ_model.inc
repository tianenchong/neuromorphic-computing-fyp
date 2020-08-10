********************************************************************************************************
********************************************************************************************************
** Title:  MTJ_model.inc
** Author: Jongyeon Kim, VLSI Research Lab @ UMN
** Email:  kimx2889@umn.edu
**Modified by: Ibrahim Ahmed, ahmed589@umn.edu 

** Update: Aug 22, 2016
********************************************************************************************************
** This library includes LLG_solver.inc, Resistance.inc, Thermal_fluctuation.inc, and Heat_diffusion.inc.
** At the given voltage across the MTJ, dynamic current and resistance are generated.
********************************************************************************************************
.include 'LLG_solver.inc'
.include 'Resistor.inc'
.include 'HeatDF.inc'


.subckt MTJ e1 e2 e3  lx='65n' ly='65n' lz='1.48n' tox='1n' Ms0='1075' P0='0.715' alpha='0.01' Tmp0='300' RA0='5.4' MA='1' ini='1' Kp='1e7' lxshm ='60n' lyshm= '40n' lzshm= '2.2n' Kp='1e7' SHA='0.4' H_app = '1200' UI = '0' InA = ' 0.01' x_ad = '0'

XRA   ex2 e2 Mx My Mz Tmp thi RA lx='lx' ly='ly' tox='tox' P0='P0' RA0='RA0' MA='MA' 
XLLG  Mx My Mz Is Istt Ias Tmp thi LLG lx='lx' ly='ly' lz='lz' Ms0='Ms0' P0='P0' alpha='alpha' MA='MA' ini='ini' lxshm ='lxshm' lyshm= 'lyshm' lzshm= 'lzshm' Kp='Kp' SHA='SHA' H_app = 'H_app' UI = 'UI' InA = ' InA'  x_ad = 'x_ad'
XHD   Ihd Tmp HD lx='lx' ly='ly' lz='lz' Tmp0='Tmp0'

*** Internal top electrode of SHM ***************
Ve1 e3 ex 0
R_shm ex e2 '2000*lxshm/(lyshm*lzshm)*1n'

*** Ishm is fed to LLG and Head_Diffusion models *********

G_shm1 0 Is cur='-I(Ve1)'


*** Internal top electrode of MTJ ***************
Ve2 e1 ex2 0


*** Charge current passing through MTJ stack *************
*** Asymetry of write current ************************
*** positive bias:Ias=Iatp, negative bias:Ias=Ipta ***
*** Iatp will generate more spin current. ************ 
.param Iatp='1'
.param Ipta='1/1.47'
E_Ias Ias 0 vol='(1+(v(e1)-v(e2))/abs(v(e1)-v(e2)))*(Iatp-Ipta)/2+Ipta'


*** Charge current passing through MTJ stack *************
*** Imtj is fed to LLG and Head_Diffusion models *********
G_Imtj1 0 Istt cur='-I(Ve2)'
G_Imtj2 0 Ihd cur='-I(Ve2)*1'

.ends
