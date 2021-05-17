// treinamento da rede neural multicamada MLP(multilayer perceptron) 
// usando o algoritmo da retropropagação do erro(error backpropagation)  
// para  o problema do OU-Exclusivo.
// função de ativação: sigmóide bipolar
// f(x)= (2/(1+exp(-x)))-1
// derivada de f(x)= 0.5(1+f(x))(1-f(x))
// data: 07/09/2020

clc;
clear;
//    tabela verdade
//       x1   x2
     x=[ 1    1  		
        -1    1   
         1   -1   
        -1   -1]; 

// vetor de saidas
     t=[ -1  
          1  
          1 
         -1 ];
     
// neurônios de entrada
neuroniosentrada=2; // duas variáveis: x1 e x2

//neurônios na camada escondida
//neuroniosescondidos=input('Neurônios na camada escondida ='); 
neuroniosescondidos= 4;
//  neurônios na camada de saída
neuroniossaida=1;

//inicialização da taxa de aprendizagem alfa
//alfa=input('Taxa de aprendizagem(0< alfa <=1) =');
alfa= 0.03;
//errototaladmissivel=input('Entre com o erro total admissivel: ');
errototaladmissivel=0.001;
//numciclo=input('Entre com o número de ciclos máximo= ');
numciclo=4000;
//------inicialização aleatória dos pesos, entre -0.5 e +0.5 ---------
// da camada escondida
v=rand(neuroniosentrada,neuroniosescondidos)-0.5;
bv=rand(neuroniosescondidos,1)-0.5; //bias

deltinhainv=zeros(neuroniosescondidos,1);
deltinhav=zeros(neuroniosescondidos,1);

// da camada de saida
w=rand(neuroniosescondidos,neuroniossaida)-0.5;
bw=rand()-0.5; //bias

deltinhaw=zeros(neuroniossaida,1);

//gráfico do erro quadratico total
//xlabel('Ciclos');
//ylabel('Erro quadratico total');

//-------- treinamento da rede neural-------
disp('Iniciando treinamento ...');
ciclo=0;
errototal=10; // acumula os erros para todos os padrões de treinamento
//zin=zeros(1,neuroniosescondidos); // soma ponderada das entradas dos neurônios escondidos
//z=zeros(1,neuroniosescondidos);// saída dos neurônios escondidos

//treinamento irá parar pelo número de ciclo ou pelo errototal alcançado.
while (ciclo < numciclo) && (errototal > errototaladmissivel)
   ciclo=ciclo+1;
   errototal=0;  
   
   // -----------fase feedforward -------------- 
   // inserção de cada padrao de treinamento na entrada da rede neural
   // e cálculo das saídas z dos neurônios escondidos
   for padroes=1:4
		for j=1:neuroniosescondidos
         zin(j)= x(padroes,:)* v(:,j)+bv(j); 
         z(j) = (2/(1+%e^(-zin(j))))-1; // sigmóide bipolar
        end
      
       //%Cálculo da  saída  da rede    
     	yin=z'*w+bw;  
        y=(2/(1+%e^(-yin)))-1;
       
   // -------------fase da Retropropagação do erro----------      
   // da saida para a camada escondida
   //cálculos do deltinhaw do neurônio de saída
     deltinhaw = (t(padroes) - y)*0.5*(1+y)*(1-y);
  // cálculo dos deltaw para atualização dos pesos do neurônio de
  // saida
     for j=1:neuroniosescondidos
	   deltaw(j)= alfa*deltinhaw*z(j);
     end
        
  //cálculo das atualizações dos bias dos neurônios de saida
     deltabw=alfa*deltinhaw;
        
 // cáculo dos deltinhav da camada escondida para as atualizações dos pesos
 // dos neurônios escondidos
    for j=1:neuroniosescondidos
      deltinhav(j)=deltinhaw*w(j)*0.5*(1+z(j))*(1-z(j));
    end
        
//cálculo dos deltav para atualização dos pesos dos neurônios escondidos   
    for i=1:neuroniosentrada
      for j=1:neuroniosescondidos
       deltav(i,j)= alfa*deltinhav(j)'*x(padroes,i);
      end
    end   
        
// cálculo dos deltabv, bias dos neurônios escondidos
    for i=1:neuroniosescondidos
      deltabv(i)= alfa*deltinhav(i);
    end
            
//-------atualização dos pesos---------- 
// dos neurônios da camada de saída
    w=w+deltaw
    bw=bw+deltabw;
// dos neurônios da camada escondida
   for i=1:neuroniosentrada
     for j=1:neuroniosescondidos
       v(i,j)= v(i,j)+deltav(i,j);   
     end
   end
   for i=1:neuroniosescondidos
     bv(i)=bv(i)+deltabv(i);
   end

   // cálculo do erro total
   // é a soma de todos os erros  produzidos pelos neurônios de saida
   // para todos os padrões de treinamento
   errototal=errototal+0.5*((t(padroes)-y)^2);
   end // for padrões de entrada
    //plot(ciclo,errototal,'r.');
    erroquadraticototal(ciclo)=errototal;
end // while
plot(erroquadraticototal,'r.');
disp('Fim do treinamento');
disp('Erro quadrático final: ');
disp(errototal);
disp('Ciclos: ');
disp(ciclo);
disp('');
disp('');
disp('Teste da rede treinada');

//--------teste da rede com os padrões de treinamento ---------
    for padroes=1:4
		for j=1:neuroniosescondidos
         zin(j)= x(padroes,:)* v(:,j)+bv(j); 
         z(j) = (2/(1+%e^(-zin(j))))-1;
        end     
        //Cálculo da saída  da rede       
     	 yin=z'*w+bw;  
         y=(2/(1+%e^(-yin)))-1;
         mprintf("Target: %f   Rede treinada: %f\n",t(padroes),y);
   end
      
      
