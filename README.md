# Modelo de Triagem
Modelo de triagem serve para auxiliar a análise de crédito no momento atual com uma verificação de risco associado a quatro parâmetros principais:

1. CNAE Primário (Atividade Principal)
2. Estado (UF)
3. Natureza Jurídica
4. Prazo de pagamento

Temos um score final baseado em uma nota de 1 a 10 para cada interação do cedente com um sacado único, de forma que o risco final é atribuído da seguinte forma:

		risco_final = 0.3*risco_cedente + 0.7*risco_sacado



## ENDPOINTS

A descrição de todos os endpoints 'get' e 'post' serão descritos e explicados a seguir:


1. get -"/" Rota padrão para debug, saber se o está tudo okay.

2. post - "/triagem" rota que realiza a triagem. 

			- input: 
				- data_input = {
					"cedente": {
						"cnpj": str/int
					},
				
					"sacado":{
						"cnpj": str/int,
						"dataOperacao":"d/m/Y",
						"dataVencimento": "d/m/y"
					}
				}
			
			- output:
				-resultado = {
					"relation': str,
					"risco_ced": int,
					"risco_sac": int,
					"risco_final": int,
					"novo_cnae": tuple/list avisando se o cnae(divisao) do sacado e cedente são novos na base
					"novo_nj": tuple/list avisando se a natureza jurídica do sacado e cedente são novos na base
				}
				
				- resultado = {
					"info": 90 - prazo de pagamento negativo
				}
				- resultado = {
					"info": 91 - datas anteriores a hoje, por exemplo data de vencimento maior que data de operação ainda retornaria um prazo positivo, porém se a data de vencimento estiver anterior ao momento retorna esse erro.
				}
				- resultado = {
					"info": 92 - prazo de vencimento igual ao dia atual
				}
				- resultado = {
					"info": 100 - cnae cedente invalido ou não consta na base
				}
				- resultado = {
					"info": 101 - cnae sacado invalido ou não consta na base
				}

3. get - "/pesos/<string:which_one>" rota para visualizar como estão definidos os pesos para o parâmetro de escolha no momento.

			-input:
				- which_one: str - uf, nj, prazo ou cnae
			
			-output:
				- resultado = json file dos pesos
			
4. post - "/change_peso" rota com o objetivo de fazer a atualização dos pesos manualmente caso seja necessário.

			- input:
				- data_input = {
					"which_one": str referenciando qual parâmetro será alterado,
					"data": {
								"key": str, 
								"value": int,
									
							},
					"force": boolean (ainda não implementado, mas quando verdadeiro força a alteração dos pesos, por exemplo, quando a key a ser alterada não existe ela é forçada a existir.),
				}
				- obs: especifico para which_one = cnae:
					data:  {
								"key_0": str, cnae cedente,
								"key_1": str, cnae sacado,
								"value": int, peso da relacao aqui
							}
				
				
			- output:
					- "info": "alterado com sucesso", vai ser alterado para um valor numérico de 80
					- "info": "key nao existente", vai ser alterado para um valor numérico de 81
					- "info": "parametro invalido", vai ser alterado para um valor numérico de 82
		
5. post - "/relation_cnaes" rota com o objetivo de buscar consulta a API do IBGE (https://servicodados.ibge.gov.br/api/docs/cnae?versao=2) para trazer informações referentes às definições do CNAE.

			-input:
				- data_input = {
					"cnae_1": str - código do cedente,
					"cnae_2": str - código do sacado,
					"mode": str - completo, secao, divisao, grupo, classe ou subclasse (nível de hierarquia na busca de informações)
						default = completo
				
				}
			
			-output:
				- resultado = {
					"descricao_cedente": {
						"descricao_secao": str, 
						"descricao_divisao": str, 
						"descricao_grupo": str,
						"descricao_classe": str,
						"descricao_subclasse": str
					},
					"descricao_sacado": {
						"descricao_secao": str, 
						"descricao_divisao": str, 
						"descricao_grupo": str,
						"descricao_classe": str,
						"descricao_subclasse": str
					}
				}
				- resultado = {
					"error": 200 - parametro invalido
				}
				- resultado = {
					"error": 400 - timeout, api demorou para responder
				}
		
6. get - "/decode/<string:which_one>/acro/<string:acro>" rota com o objetivo de decodificar os acrônimos utilizados nos arquivos de pesos dos parâmetros. 

			-input:
				- which_one: string que identifica a variável desejada (cnae ou nj)
				- label: string representando o acrônimo que deseja-se decodificar
				
			-output:
				-	resultado = {
						'label_decoded': string
					}
				-	resultado = {
						'info': 85 (cnae não existente no mapa de decodificação)
					}
				-	resultado = {
						'info': 86 (nj não existente no mapa de decodificação)
					}
				-	resultado = {
						'info': 82 (parâmetro de escolha inválido)
					}
				

7. get - "/analise_cnae/divisao/<int:divisao>/how/<string:how>" rota responsável pela análise do capital envolvido na divisão do cnae atuando segundo o parâmetro *how*. Basicamente, busca-se o top 3 de interação entre a divisão dada como input pelo usuário com as demais divisões.

			-input:
				- divisao: str/int que identifica a divisão do cnae em escolha
				- how: str (cedente ou sacado) representa o modo que será analisado, por exemplo quando *how* = cedente, o top 3 é definido a partir das divisões que atuaram como sacado e tiveram relações com tal divisão de entrada. O contrário representa a interação de uma divisão que atuou como sacado com as divisões que foram seu cedente.
			
			-output:
				resultado = {
					divisao_top_1: {
						valor_total: float,
						valor_atraso: float
					},
					divisao_top_2: {
						valor_total: float,
						valor_atraso: float
					},
					divisao_top_3: {
						valor_total: float,
						valor_atraso: float
					}
					
				}
				
8. get - "/comp/divisao/<int:divisao>" rota responsável pela comparação entre a performance de uma divisão atuando como cedente versus atuando como sacado. 
	
			-input:
				- divisao: str/int que identifica a divisão do cnae em escolha
			
			-output:
				- resultado = {
					'as_ced': float,
					'as_sac': float,
					'melhor': string
				}
				- resultado = {
					'info': 130 (Sem informações suficientes para realizar a comparação, ou seja, sem dados tanto como cedente quanto como sacado)
				}
				- resultado = {
					'info': 132, (Sem informações suficientes para ver a performance como sacado, ou seja, sem dados da divisão atuando como sacado)
					'as_ced': float
				}
				- resultado = {
					'info': 131, (Sem informações suficientes para ver a performance como cedente, ou seja, sem dados da divisão atuando como cedente)
					'as_sac': float
				}
				
		
	# Exemplo em Python para a rota 2:
	
			import requests
			import json
			url = 'http://localhost/triagem'
			headers = {'content-type': 'application/json'}
			data_input = {
				'cedente': {
					'cnpj': '78466141000272',
        
				},
				'sacado':{
					'cnpj': '18758944000198',
					'dataOperacao': '02/04/2020',
					'dataVencimento': '28/04/2020'
				}
			}
			
			r = requests.post(url = url, data = json.dumps(data_input), headers=headers)
			data = r.json()
			print(data)
			
			{
				'novo_cnae':[1, 1],
				'novo_nj': [0, 0],
				'relation': '29 -> 27',
				'risco_ced': 6,
				'risco_final': 6,
				'risco_sac': 6
			}
	
	
	# Exemplo em Python para a rota 3:
		
			url_2 = 'http://localhost/pesos/uf'
			r = requests.get(url = url_2)
			data = r.json()
			print(data)
			{
				'AL': 2,
				'AM': 6,
				'BA': 6,
				'CE': 7,
				'DF': 8,
				'ES': 4,
				'GO': 6,
				'MA': 5,
				'MG': 6,
				'MS': 3,
				'MT': 2,
				'PA': 3,
				'PB': 7, 
				'PE': 7,
				'PI': 7, 
				'PR': 7, 
				'RJ': 8, 
				'RN': 4,
				'RO': 3,
				'RR': 2,
				'RS': 6,
				'SC': 7,
				'SE': 3,
				'SP': 9, 
				'TO': 2, 
				'date': '2020-04-02 13:39:07.815279'
			}
			
	# Exemplo em Python para a rota 4:
	
			data = {'which_one':'uf', 'data': {'key': 'RS',  'value': 6}}

			url_3 = 'http://localhost/change_peso'
			r = requests.post(url = url_3, data = json.dumps(data), headers = headers)
			data = r.json()
			print(data)
				True
	
	# Exemplo em Python para a rota 5:
	
			url_4 = 'http://localhost/relation_cnaes'
			data = {
				"cnae_1":"74.10-2-03",
				"cnae_2":"93.13-1-00",
				"mode": "secao"
			}
			r = requests.post(url = url_4, data = json.dumps(data), headers = headers)
			print(r.json())
			{
				'descricao_cedente': {
					'descricao_secao': 'ATIVIDADES PROFISSIONAIS, CIENTÍFICAS E TÉCNICAS'
				},
				
				'descricao_sacado': {
					'descricao_secao': 'ARTES, CULTURA, ESPORTE E RECREAÇÃO'}
				}
	
			}
	
	# Exemplo em Python para a rota 6:
		
			url_5 = 'http://localhost/decode/{}/acro/{}'.format('nj', 'NJ2')
			r = requests.get(url = url_5)
			data = r.json()
			print(data)
			{
				'label_decoded': '230-5 - Empresa Individual de Responsabilidade Limitada (de Natureza Empresária)'
			}
			
	
	# Exemplo em Python para a rota 7:
	
			url_6 = 'http://localhost/analise_cnae/divisao/{}/how/{}'
			r = requests.get(url=url_6.format(28, "cedente"))
			print(r.json())			
			{
				'46': {
					'valor_atraso': 94145.93,
					'valor_total': 649892.2399999999
				},
				'47': {
					'valor_atraso': 115952.47999999998,
					'valor_total': 1079683.5999999999
				},
				'70': {
				'valor_atraso': 0.0, 
				'valor_total': 1928002.08
				}
			} 
	# Exemplo em Python para a rota 8:
	
			url_7 = 'http://localhost/comp/divisao/{}'
			r = requests.get(url = url_7.format(28))
			print(r.json())
			{
				'as_ced': 94.256,
				'as_sac': 100.0,
				'melhor': 'sacado'
			}