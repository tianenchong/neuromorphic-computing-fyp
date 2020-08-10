************************************************************************************
************************************************************************************
** Title:  MTJ_write.sp
** Author: Ibrahim Ahmed, VLSI Research Lab @ UMN
** Email:  ahmed589@umn.edu
************************************************************************************
** This run file simulates the dynamic motion of  MTJ.
** # Instruction for simulation
** 1. Set the MTJ dimensions and material parameters.
** 2. Select anisotropy(IMA/PMA) and initial state of free layer(P/AP).
** 3. Adjust bias voltage for Read/Write operation.
** ex. APtoP switching: positive voltage @ ini='1'
**     PtoAP switching: negative voltage @ ini='0'  
************************************************************************************
** # Description of parameters
** lx,ly,lz: width, length, and thickness of free layer
** tox: MgO thickness
** Ms0:saturation magnetizaion at 0K
** P0: polarization factor at 0K 
** alpha: damping factor
** temp: temperature
** MA: magnetic anisotropy (MA=0:In-plane,MA=1:Perpendicular)
**     also sets magnetization in pinned layer, MA=0:[0,1,0],MA=1:[0,0,1]
** ini: initial state of free layer (ini=0:Parallel,ini=1:Anti-parallel)
**  This model can be used for 4 different switching mechanism: 1) STT Only, 2) SHE Only + Ext. field, 3) STT + SHE, 4) STT + SHE + Ext. field
** External field is defined in 'H_app' in Oe
** UI = 1 for user defined initial angle, 'InA', UI = 0 for avg. initial angle
** 'x_ad' is the additional field like torque. It  is defined as a fraction of damping like torque. 
** So, x_ad = 0.5 means the field like torque has a magnitude half of the damping like torque


** The following code is the switching mechanism type 2)SHE only switching with external field
************************************************************************************

.include 'MTJ_model.inc'

*** Options ************************************************************************
.option post
.save

*** Voltage biasing to MTJ *********************************************************
.param RAp=5
.param t = 5e-9

.param istt = '00u' 
.param ISHM = '400u' 
.param tshm  = '800p'

.param damping = 0.02

I1 1 0 PULSE (0 ISTT  1.5n 100p 100p 50n 80n)   
I2 3 0 PULSE (0 ISHM 1.5n 100p 100p tshm 80n)  

V_2 2 0 0

XMTJ1 1 2 3 MTJ lx='45n' ly='45n' lz='0.7n' tox='1n' Ms0='1185'  P0='0.73' alpha='0.02' Tmp0='300' RA0='RAp' MA='1' ini='0' Kp='9e6'  lxshm ='60n' lyshm= '45n' lzshm= 't' SHA='0.4' H_app = ' 400 ' UI = '1' InA = ' 0.02436'  x_ad ='0'

.param pw='30ns' 
.tran 2p 'pw' START=2.5e-18 uic 

.end
