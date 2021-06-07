import cmath
import numpy as np
import random
from matplotlib import pyplot as plt 
from numpy import mean

#num total de lankmarks
N=10
#declaração inicial dos vetorews
x=np.zeros(N)
y=np.zeros(N)
vx=[]
vy=[]
#vetor de angulos de aberturas com relacao a z0
phi=[]
#vetor medido de angulo absoluto
tau=np.zeros(N)
#distancias medidas ate os landmarks
MD=[]
#posição dos landmarks medidos
X=[]
Y=[]
#poses estimadas
PEx=[]
PEy=[]
#pose inicial
pose=[0,0,0]
estimated_pose=[0,0,0]
size=100

##funcoes
def land_generator(N,size):
  for i in range (N):
    x[i]=random.randint(0,size)
    y[i]=random.randint(0,size)
    #z[i]=complex(x[i],y[i])
  return x,y

def distance_calculator(x1,y1,x2,y2):
  return np.sqrt((x1-x2)**2+(y1-y2)**2)
  
def distance_to_z0(x,y,n,vx,vy):
  for i in range (n):
    vx.append(x[i]-x[0])
    vy.append(y[i]-y[0])
  return vx,vy
  
def gera_pose_robo():
  x=random.randint(0,size)
  y=random.randint(0,size)
  theta=np.deg2rad(15*random.randint(0,23))
  return x,y,theta
  
def tau_calculator(pose,x,y,N):
  #vetor de orientação do rodo
  xr=np.cos(pose[2])
  yr=np.sin(pose[2])
  for i in range (N):
    #posicao do landmark no referencial do robo
    xir=x[i]-pose[0]
    yir=y[i]-pose[1]
    #tau com relação ao ^x
    tau[i]=cmath.phase(complex(xir,yir))-cmath.phase(complex(xr,yr))
  return tau    

#ajustar
def realiza_medidas(pose,x,y,tau,X,Y,MD,phi,N):
  cont=0
  glob=1
  tau0=0
  erroR=glob*0.1
  erroA=glob*np.deg2rad(5)
  xr=np.cos(pose[2])
  yr=np.sin(pose[2])
  theta=cmath.phase(complex(xr,yr))
  for i in range (N):
    if np.abs(tau[i])<=np.pi/2 or np.abs(tau[i])>=3*np.pi/2:
      if cont==0:
        zx0=x[i]
        zy0=y[i]
        tau0=tau[i]
      X.append(x[i])
      Y.append(y[i])      
      dist=distance_calculator(x[i],y[i],pose[0],pose[1])*(1+erroR*random.uniform(-1.0,1.0))
      if dist<0:
        dist=0
      MD.append(dist)
      phi.append((tau[i]-tau0)+erroA*random.uniform(-1.0,1.0))
      cont=cont+1      
  return X, Y, MD, phi,cont,tau0
  
  
def pose_calculator(X,Y,phi,MD,vx,vy,n,tau0):
  z0m=complex(0.0,0.0)
  if n<2:
    print("Poucas medidas realizadas")
    estimated_pose=[0,0,0]
  else:
    print(n," medidas realizadas")
    for i in range (1,n):
      vi=complex(vx[i],vy[i])
      z0=vi/((MD[i]/MD[0])*np.exp(phi[i]*1j)-1)
      z0m=z0m+z0
      PEx.append((complex(X[0],Y[0])-z0).real)
      PEy.append((complex(X[0],Y[0])-z0).imag)
      #print("contas:",z0,z0e,phi[i])
    z0m=z0m/n
    thetaestimado=cmath.phase(z0m)-tau0
    if thetaestimado<0:
      thetaestimado=thetaestimado+2*np.pi
    estimated_pose=[mean(PEx),mean(PEy),thetaestimado]      
  return PEx,PEy,estimated_pose
    
def plot_map(PEx,PEy,cont,x,y,X,Y,pose,N):
  if cont>1:
    #plt.plot(PEx, PEy,'.',label='Posições estimadas robô',color='tab:gray')
    plt.plot(estimated_pose[0],estimated_pose[1],'.',label='Posição média estimada',color='r',markersize=10)
    print("Pose real    : [","{:.4f}".format(pose[0]),"{:.4f}".format(pose[1]),"{:.2f}".format(np.rad2deg(pose[2])),"]")
    print("Pose estimada: [","{:.4f}".format(estimated_pose[0]),"{:.4f}".format(estimated_pose[1]),"{:.2f}".format(np.rad2deg(estimated_pose[2])),"]")
    
  plt.plot(x, y,'*',label='Landmarks',markersize=10,color='tab:orange')
  plt.plot(X, Y,'.',color='b',label='Landmarks medidos')
  plt.plot(pose[0], pose[1],'.',label='Posição real robô',color='k')
  for i in range(N):
      plt.text(x[i], y[i], str(i))
  
  origin = np.array([[pose[0]],[pose[1]]]) # origin point
  widths = np.linspace(0, 40, origin.size)
  plt.quiver(*origin, np.cos(pose[2]), np.sin(pose[2]), color=['y'],label='orientação robô',width=0.005)
  A=10
  plt.xlim([-2*A, size+2*A])
  plt.ylim([-A, size+A])
  plt.legend()
  plt.show()
  
def erro_medio(pe,p,cont):
  erro=-1
  if cont>1:  
    erro=distance_calculator(pe[0],pe[1],p[0],p[1])
    print("Erro: ",erro)
  return erro
    
########## MAIN ##########
#gera posições dos landmarks
x,y=land_generator(N,size)
#gera pose do robo
pose=gera_pose_robo()
#calcula os taus
tau=tau_calculator(pose,x,y,N)
#realiza medidas com os sensores
X, Y, MD, phi,cont,tau0=realiza_medidas(pose,x,y,tau,X,Y,MD,phi,N)
#para cada lankmark, calcula os vetores vi=zi-z0
vx,vy=distance_to_z0(X,Y,cont,vx,vy)
#faz a estimativa da pose
PEx,PEy,estimated_pose=pose_calculator(X,Y,phi,MD,vx,vy,cont,tau0)
#calcula o erro na distancia
erro=erro_medio(estimated_pose,pose,cont)
#faz o grafico
plot_map(PEx,PEy,cont,x,y,X,Y,pose,N)





