import pickle
import json
import urllib.request

def main():
    #Get symbols with baseAsset and quoteAsset for all symbols in response
    response = urllib.request.urlopen("https://api.binance.com/api/v3/exchangeInfo").read()
    response = json.loads(response)
    symbols = []
    for symbol in response['symbols']:
        symbols.append((symbol['symbol'], symbol['baseAsset'], symbol['quoteAsset']))

    #Transform symbols into a dictionary with symbol as key and baseAssets and quoteAsset as keys for a nested dictionary
    symbols_dict = {}
    for symbol in symbols:
        symbols_dict[symbol[0]] = {'baseAsset': symbol[1], 'quoteAsset': symbol[2]}

    priority_list_order = ['USDT', 'USD', 'USDC', 'BUSD', 'BTC', 'ETH', 'BNB']

    #Make a new dictionary where the keys are the baseAssets and the values are a list of the quoteAssets
    base_assets = {}
    for key in symbols_dict:
        if symbols_dict[key]['baseAsset'] not in base_assets:
            base_assets[symbols_dict[key]['baseAsset']] = [symbols_dict[key]['quoteAsset']]
        else:
            base_assets[symbols_dict[key]['baseAsset']].append(symbols_dict[key]['quoteAsset'])

    #Keep only values in new_base_assets that are in priority_list_order
    for key in base_assets:
        base_assets[key] = [x for x in base_assets[key] if x in priority_list_order]

    #Dump new_base_assets in a pickle file in one line
    pickle.dump(base_assets, open('base_assets.pkl', 'wb'))

if __name__ == '__main__':
    main()