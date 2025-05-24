# ShallowNN vs. Baseline
## Baseline
25.8%

                      precision    recall  f1-score   support

Art Nouveau (Modern)     0.0000    0.0000    0.0000         0
             Baroque     0.6364    0.0700    0.1261       100
       Expressionism     0.3605    0.3100    0.3333       100
       Impressionism     0.3165    0.4237    0.3623        59
  Post-Impressionism     0.1818    0.1463    0.1622        41
              Rococo     0.0000    0.0000    0.0000         0
         Romanticism     0.0000    0.0000    0.0000         0
          Surrealism     0.3797    0.6000    0.4651       100
           Symbolism     0.0000    0.0000    0.0000       100

            accuracy                         0.2580       500
           macro avg     0.2083    0.1722    0.1610       500
        weighted avg     0.3276    0.2580    0.2410       500

## Shallow NN
29%

Art Nouveau (Modern)     0.0000    0.0000    0.0000         0
             Baroque     0.4615    0.0600    0.1062       100
       Expressionism     0.3333    0.4700    0.3900       100
       Impressionism     0.3204    0.5593    0.4074        59
  Post-Impressionism     0.2619    0.2683    0.2651        41
              Rococo     0.0000    0.0000    0.0000         0
         Romanticism     0.0000    0.0000    0.0000         0
          Surrealism     0.4019    0.4300    0.4155       100
           Symbolism     0.2632    0.0500    0.0840       100

            accuracy                         0.2900       500
           macro avg     0.2269    0.2042    0.1854       500
        weighted avg     0.3513    0.2900    0.2690       500

# Hierarchy
## Hierarchy
29.8%

                      precision    recall  f1-score   support

Art Nouveau (Modern)     0.0000    0.0000    0.0000         0
             Baroque     0.7778    0.0700    0.1284       100
       Expressionism     0.3981    0.4100    0.4039       100
       Impressionism     0.3614    0.5085    0.4225        59
  Post-Impressionism     0.3056    0.2683    0.2857        41
         Romanticism     0.0000    0.0000    0.0000         0
          Surrealism     0.4069    0.5900    0.4816       100
           Symbolism     0.0588    0.0100    0.0171       100

            accuracy                         0.2980       500
           macro avg     0.2886    0.2321    0.2174       500
        weighted avg     0.3960    0.2980    0.2795       500

## 4-patch
27.2%

                      precision    recall  f1-score   support

Art Nouveau (Modern)     0.0000    0.0000    0.0000         0
             Baroque     0.7273    0.0800    0.1441       100
       Expressionism     0.2941    0.3500    0.3196       100
       Impressionism     0.3243    0.6102    0.4235        59
  Post-Impressionism     0.2083    0.2439    0.2247        41
              Rococo     0.0000    0.0000    0.0000         0
         Romanticism     0.0000    0.0000    0.0000         0
          Surrealism     0.4369    0.4500    0.4433       100
           Symbolism     0.1667    0.0200    0.0357       100

            accuracy                         0.2720       500
           macro avg     0.2397    0.1949    0.1768       500
        weighted avg     0.3803    0.2720    0.2570       500

## 2-patch(Main Object)
                      precision    recall  f1-score   support

Art Nouveau (Modern)     0.0000    0.0000    0.0000         0
             Baroque     0.7500    0.0900    0.1607       100
       Expressionism     0.3115    0.3800    0.3423       100
       Impressionism     0.3394    0.6271    0.4405        59
  Post-Impressionism     0.2308    0.2927    0.2581        41
              Rococo     0.0000    0.0000    0.0000         0
         Romanticism     0.0000    0.0000    0.0000         0
          Surrealism     0.4123    0.4700    0.4393       100
           Symbolism     0.0000    0.0000    0.0000       100

           micro avg     0.2860    0.2860    0.2860       500
           macro avg     0.2271    0.2066    0.1823       500
        weighted avg     0.3537    0.2860    0.2616       500


## 6-patch
                      precision    recall  f1-score   support

Art Nouveau (Modern)     0.0000    0.0000    0.0000         0
             Baroque     0.6364    0.0700    0.1261       100
       Expressionism     0.3158    0.4200    0.3605       100
       Impressionism     0.3231    0.7119    0.4444        59
  Post-Impressionism     0.2683    0.2683    0.2683        41
              Rococo     0.0000    0.0000    0.0000         0
         Romanticism     0.0000    0.0000    0.0000         0
          Surrealism     0.4352    0.4700    0.4519       100
           Symbolism     0.0000    0.0000    0.0000       100

            accuracy                         0.2980       500
           macro avg     0.2199    0.2156    0.1835       500
        weighted avg     0.3376    0.2980    0.2622       500

## 2-patch more data
                      precision    recall  f1-score   support

Art Nouveau (Modern)     0.0000    0.0000    0.0000         0
             Baroque     0.7333    0.1100    0.1913       100
       Expressionism     0.3514    0.3900    0.3697       100
       Impressionism     0.4306    0.5254    0.4733        59
  Post-Impressionism     0.3023    0.3171    0.3095        41
              Rococo     0.0000    0.0000    0.0000         0
         Romanticism     0.0000    0.0000    0.0000         0
          Surrealism     0.3688    0.5200    0.4315       100
           Symbolism     0.0625    0.0300    0.0405       100

            accuracy                         0.2980       500
           macro avg     0.2499    0.2103    0.2018       500
        weighted avg     0.3788    0.2980    0.2878       500



        