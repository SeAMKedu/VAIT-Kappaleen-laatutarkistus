[![DOI](https://zenodo.org/badge/1465376099.svg)](https://doi.org/10.5281/zenodo.24396263)

![Älykkäät teknologiat](https://storage.googleapis.com/seamk-production/2022/04/2b1d63e0-alykkaat-teknologiat_highres_2022-768x336.jpg)
![EPLiitto logo](https://github.com/SeAMKedu/VAIT-tekoaly-rag/blob/main/kuvat/EPLiitto_logo_vaaka_vari.jpg)
![EU osarahoitus](https://github.com/SeAMKedu/VAIT-tekoaly-rag/blob/main/kuvat/FI_Co-fundedbytheEU_RGB_POS.png)

# Mistral-kameratarkastus

Kamerapohjainen laaduntarkastussovellus, joka tunnistaa tuotteen tarkastusalueelta ja luokittelee sen Mistralin konenäkömallilla.

Sovellus näyttää kamerakuvan Tkinter-käyttöliittymässä. Käyttäjä rajaa kuvasta tarkastusalueen eli ROI:n, minkä jälkeen sovellus tunnistaa tuotteen saapumisen, odottaa sen pysähtymistä ja lähettää rajatun kuvan Mistralin API:lle tarkastettavaksi.

## Ominaisuudet

- Käyttäjän määrittämä tarkastusalue kamerakuvasta
- Tyhjän tarkastusalueen kalibrointi
- Tuotteen liikkeen ja pysähtymisen tunnistus
- Kuvan rajaus ennen pilvipalveluun lähettämistä
- PASS- ja FAIL-luokittelu syykoodeineen
- Epävarman tai virheellisen mallivastauksen hylkääminen
- Päivittäiset CSV- ja tekstilokit
- FAIL-kuvien ja otantaan valittujen PASS-kuvien tallennus
- Päivittäisten tarkastusmäärien ja hylkäyssyiden seuranta

## Vaatimukset

- Python 3.11 tai uudempi
- OpenCV:n tukema kamera (Normaali webkamera yleensä toimii)
- Mistral API -avain
- Mistral-tilillä käyttöoikeus valittuun konenäkömalliin

## Asennus

Luo virtuaaliympäristö ja asenna riippuvuudet:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## API-avaimen asettaminen

Sovellus lukee avaimen ympäristömuuttujasta `MISTRAL_API_KEY`. Älä tallenna oikeaa API-avainta lähdekoodiin, `.env.example`-tiedostoon tai Git-repositorioon.

PowerShellissä avaimen voi antaa nykyiselle istunnolle näin ilman, että itse avain tallentuu komentohistoriaan:

```powershell
$env:MISTRAL_API_KEY = Read-Host "Mistral API -avain"
```

Käynnistä sovellus:

```powershell
python .\mistral_inspector.py
```

## Käyttö

1. Rajaa tarkastettava alue vetämällä suorakulmio kamerakuvan päälle.
2. Varmista, että tarkastusalue on tyhjä ja normaalissa odotustilassa.
3. Valitse **Set EMPTY ARM Baseline**.
4. Tuo tarkastettava tuote alueelle.
5. Sovellus odottaa tuotteen pysähtymistä ja käynnistää tarkastuksen automaattisesti.
6. Poista tuote alueelta ennen seuraavaa tarkastusta.

## Asetukset

Keskeiset asetukset ovat `mistral_inspector.py`-tiedoston alussa. Niillä voi säätää esimerkiksi:

- kameran tunnistetta ja resoluutiota;
- liikkeen ja tyhjän alueen tunnistuskynnyksiä;
- pysähtymiseen vaadittavaa kuvamäärää;
- tarkastusten välistä jäähdytysaikaa;
- mallille lähetettävän kuvan enimmäiskokoa ja JPEG-laatua;
- PASS-kuvien tallennusväliä.

Oletusmalli on `mistral-large-4-0`. Sen voi vaihtaa ilman lähdekoodin muokkaamista ympäristömuuttujalla:

```powershell
$env:MISTRAL_MODEL_ID = "tuettu-konenakomalli"
```

Tarkista ennen vaihtamista Mistralin dokumentaatiosta, että malli tukee kuvia ja JSON-muotoisia vastauksia.

## Tietosuoja ja tallennettavat tiedot

Sovellus lähettää käyttäjän rajaaman kamera-alueen Mistralin Chat Completions API:lle. Älä sijoita rajatulle alueelle henkilöitä, henkilötietoja, tunnuksia, luottamuksellisia asiakirjoja tai tarkastukseen kuulumatonta tuotantotietoa.

Mistralin tietosuojakäytännön mukaan API-syötteitä ja vastauksia voidaan normaalisti säilyttää väärinkäytösten valvontaa varten. Zero Data Retention -asetuksella tuettujen API-kutsujen säilytystä voidaan rajoittaa. Tarkista voimassa olevat ehdot ja oman organisaation asetukset ennen tuotantokäyttöä:

- [Mistralin tietosuojakäytäntö](https://legal.mistral.ai/terms/privacy-policy/)
- [Zero Data Retention](https://docs.mistral.ai/admin/monitor-comply/zero-data-retention)

Sovellus tallentaa paikallisesti:

- kaikki hylätyt tarkastuskuvat;
- oletuksena joka 20. hyväksytyn tarkastuksen kuvan;
- CSV- ja tekstimuotoiset tarkastuslokit;
- sovelluksen teknisen lokin.

Tiedot tallennetaan `inspection_logs/`-hakemistoon ja `production_mistral_lib.log`-tiedostoon. Ne on rajattu Gitin ulkopuolelle `.gitignore`-tiedostolla. Käyttäjä vastaa tiedostojen käyttöoikeuksista, säilytysajasta, varmuuskopioinnista ja poistamisesta.

## Rajoitukset

Yleiskäyttöinen konenäkömalli voi tehdä virheitä. Testaa tunnistus riittävän laajalla ja todellista käyttöympäristöä vastaavalla aineistolla. Älä käytä sovellusta turvallisuuskriittisen tai sopimuksellisen laatupäätöksen ainoana perusteena ilman asianmukaista validointia ja muita varmistuksia.

Kamerakohtaiset valotus-, tarkennus- ja väriasetukset voivat vaikuttaa tuloksiin. Kaikki kamerat eivät myöskään noudata ohjelmallisesti pyydettyä resoluutiota tai muita asetuksia.

## Projektin tiedostot

```text
mistral-inspector/
├── mistral_inspector.py
├── requirements.txt
├── README.md
├── LICENSE
```

## Lisenssi

Projektin lähdekoodi on julkaistu [MIT-lisenssillä](LICENSE). Riippuvuudet asennetaan erikseen PyPI:stä niiden omilla lisensseillä. 
