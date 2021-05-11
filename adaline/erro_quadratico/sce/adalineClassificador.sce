// implementação da Regra Delta para treinamento do adaline
// aplicação: classificação de padrões
// data: 24/08/2020
// autor: Keiji Yamanaka

clc; // limpa area de comando
clear;// limpa as variáveis da area de trabalho

// dados de treinamento:
// 
//       x1         x2
x= [ 2.2156048   2.0636896
     0.2249505   1.5866494
     0.2949678   0.6510604     
     2.3276795   2.9326247     
     2.4975727   2.3221797
     0.1694251   1.943028     
     1.2747555   2.4284981 
     1.5264317   0.5969313
     2.0095288   2.1610245
     1.7594947   0.3425176
     1.3677854   0.9389618     
     2.1734127   2.7195418
     0.856965    1.9043969     
     2.2102014   1.868964 
     1.5877021   1.6424338
     0.3502761   0.8402314
     1.4416383   0.0909778
     0.1857691   1.3279701
     2.7648774   1.1492552
     1.9472657   1.5981004];
     
t=[ -1  1  1 -1 -1  1 -1  1 -1  1  1 -1  1 -1 -1  1  1  1 -1 -1];

// gera gráfico dos pontos
clf(); // limpa a janela de gráficos
set(gca(),"auto_scale", "on");
set(gca(),"data_bounds", [0,0;4,4]);
title("Dados");
xlabel("x1");
ylabel("x2");
da=gda();
da.y_location="origin";
da.x_location="origin";
for ponto=1:20
    if t(ponto)==1
        plot(x(ponto,1), x(ponto,2),'bd');
    else
        plot(x(ponto,1), x(ponto,2),'rd');
    end
end

// inicialização das variáveis e dos parâmetros
want=0.5- rand(1,2,"uniform"); // inicialização dos pesos 
                               // entre -0.5 e +0.5
bant=0.5- rand();
teta = 0;// limiar  da função de ativação degrau(rede treinada)
alfa = 0.01; // taxa de aprendizagem
numciclos=5000;  // número  total de ciclos de treinamento
ciclos=0;  // conta o número de vezes que os dados foram apresentados


// treinar enquanto condição de parada não for satisfeita
mprintf("Treinamento do Adaline\n");
while ciclos<=numciclos // limite de treinamento
    erroquadratico=0;
    ciclos=ciclos+1; //  conta número de ciclos de treinamento
    mprintf("ciclos = %d\n", ciclos);
     for entrada =1:20 // apresenta todos os padrões de entrada
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
for entrada =1:20 // apresenta todos os padrões de entrada
    yliquido = wnovo(1)*x(entrada,1)+ wnovo(2)*x(entrada,2)+bnovo;
    // função de ativação para o teste: degrau
    if yliquido >= teta
      y=1;
    else
      y=-1;
    end
    mprintf('t(%d)= %d   y(%d):  %d\n',entrada, t(entrada),entrada, y);
end

// imprimindo a fronteira de separação
for abcissa=-0.5:0.1:3.5
    ordenada = (-abcissa*wnovo(1)-bnovo)/wnovo(2);
    plot(abcissa, ordenada,'g.') ;
end

