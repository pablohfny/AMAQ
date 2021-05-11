// implementação da Regra Delta para treinamento do adaline
// aplicação: funções lógicas  bipolares
// data: 24/08/2020
// autor: Keiji Yamanaka

clc; // limpa area de comando
clear;// limpa as variáveis da area de trabalho

// dados de treinamento
// tabela verdade
//  x1  x2
x=[ 1  1   
   -1  1
    1 -1
   -1 -1]; 
t=[1  1  1 -1];

// gera gráfico dos pontos
clf(); // limpa a janela de gráficos
set(gca(),"auto_scale", "on");
set(gca(),"data_bounds", [-2,-2;2,2]);
title("Dados");
xlabel("x1");
ylabel("x2");
da=gda();
da.y_location="origin";
da.x_location="origin";
for ponto=1:4
    if t(ponto)==1
        plot(x(ponto,1), x(ponto,2),'bd');
    else
        plot(x(ponto,1), x(ponto,2),'rd');
    end
end


//set(gca(),"data_bounds", [-2,-2;2,2]);
//title("Curva do Erro quadrático total");
//xlabel("Ciclos");
//ylabel("Erro quadrático total");
//da=gda();
//da.y_location="origin";
//da.x_location="origin";



/// ------treinamento do adaline ------

// inicialização das variáveis e dos parâmetros
want=0.5- rand(1,2,"uniform"); // inicialização dos pesos
bant=0.5- rand();
teta = 0;// limiar  da função de ativação degrau(rede treinada)
alfa = 0.1; // taxa de aprendizagem(0<alfa<=1)
numciclos=500;  // número  total de ciclos de treinamento
ciclos=0;  // conta o número de vezes que os dados foram apresentados

// treinar enquanto condição de parada não for satisfeita
mprintf("Treinamento do Adaline\n");
while ciclos<=numciclos // limite de treinamento
    erroquadratico=0; // cálculo do erro quadratico total
    ciclos=ciclos+1; //  conta número de ciclos de treinamento
    mprintf("ciclos = %d\n", ciclos);
     for entrada =1:4 // apresenta todos os padrões de entrada
         yliquido = want(1)*x(entrada,1)+ want(2)*x(entrada,2)+bant;
         // função de ativação: linear
         y=yliquido; 
         // cálculo do erro quadrático
         erroquadratico= erroquadratico+(t(entrada)-y)^2;
         // atualização dos pesos
         wnovo(1)= want(1)+alfa*(t(entrada)-y)*x(entrada,1);
         wnovo(2)= want(2)+alfa*(t(entrada)-y)*x(entrada,2);
         bnovo=bant+alfa*(t(entrada)-y);
         // salva os pesos para a próxima atualização
         want=wnovo;
         bant=bnovo;
         
     end
     //plot(ciclos, erroquadratico, 'r*');
     
end

// ----- teste da rede treinada ------
mprintf('Teste da rede treinada\n\n');
for entrada =1:4 // apresenta todos os padrões de entrada
    yliquido = wnovo(1)*x(entrada,1)+ wnovo(2)*x(entrada,2)+bnovo;
     //função de ativação para o teste: degrau
    if yliquido >= teta
      y=1;
    else
      y=-1;
    end
    mprintf('t(%d)= %d   y(%d):  %d\n',entrada, t(entrada),entrada, y);
   //mprintf('t(%d)= %d   y(%d):  %f\n',entrada, t(entrada),entrada, yliquido);

end

// imprimindo a fronteira de separação
for abcissa=-2:0.1:2
    ordenada = (-abcissa*wnovo(1)-bnovo)/wnovo(2);
    plot(abcissa, ordenada,'g.') ;
end
