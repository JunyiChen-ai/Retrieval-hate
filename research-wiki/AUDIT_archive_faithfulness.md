# AUDIT: Archive faithfulness (MHC / MHC_zh, Qwen2.5-VL-7B archive)

> **模型辅助审计,终审留给人类。** 本表由 Claude 对照原始 title+transcript 逐条给出初判 (faithful / hallucination / whitewash / insufficient-evidence),所有判定均待人工抽查确认后方可引用。

## 方法

- 样本: 60 条,EN (MHC) / ZH (MHC_zh) 各 30,按原始 MultiHateClip 三分类 (Hateful / Offensive / Normal) 各 10 条分层随机抽样 (seed=0);W2 疑似标签噪声样本 `XScP1AiMkNM`(牛油果酱)与 `QvPp8Q7QhWE`(数钱)强制纳入。
- 对照材料: 原始 `annotation(new).json` 的 Title + Transcript(转写超过 1500 字符时截断展示,长度已标注)与二值 gt 标注 (harmful=Hateful∪Offensive)。
- 核查项: (1) `target_groups` 是否有文本依据; (2) `mechanism` 是否与转写内容相符; (3) 幻觉 = 档案声称 speech/on_screen_text 中存在转写里不存在的内容,或凭空给出无任何依据的目标/机制; (4) 洗白 = 文本毒性明显 (label=Hateful/Offensive 且转写可见毒性) 但档案输出 explicitness=none / 空 mechanism / benign summary。
- 重要限制: 档案由 MLLM 看 **视频帧+转写** 生成,而本审计只能看到 **文本**。纯视觉断言(如 visual cue、画面描述)无法据文本证实或证伪,只要不与文本矛盾就不计为幻觉,并在备注中标注 `visual-unverifiable`。文本无毒但 label=harmful 的条目(毒性可能在画面/语音语调里,或本身是标签噪声)判 `insufficient-evidence`。

## 汇总

- **EN** (n=30): faithful 23 (77%) / hallucination 3 (10%) / whitewash 2 (7%) / insufficient-evidence 2 (7%)
- **ZH** (n=30): faithful 23 (77%) / hallucination 6 (20%) / whitewash 1 (3%) / insufficient-evidence 0 (0%)
- **ALL** (n=60): faithful 46 (77%) / hallucination 9 (15%) / whitewash 3 (5%) / insufficient-evidence 2 (3%)

| 语言 | 三分类 | n | faithful | hallucination | whitewash | insufficient-evidence |
|---|---|---|---|---|---|---|
| English | Hateful | 10 | 7 | 0 | 2 | 1 |
| English | Offensive | 10 | 9 | 0 | 0 | 1 |
| English | Normal | 10 | 7 | 3 | 0 | 0 |
| Chinese | Hateful | 10 | 10 | 0 | 0 | 0 |
| Chinese | Offensive | 10 | 9 | 0 | 1 | 0 |
| Chinese | Normal | 10 | 4 | 6 | 0 | 0 |

- target 字段有依据: 6/7 (另 53 条 target 为空或不可核验,记 n.a.)
- mechanism 与内容相符: 38/53 (另 7 条 mechanism 为空或不可核验,记 n.a.)

## 关键发现(模型初判,待人工确认)

1. **幻觉几乎全是"字段级虚报",不是内容编造。** 9 例 hallucination 中 8 例是 benign 内容(医疗科普、儿歌、祝福、测评)被安上无依据的 `mechanism`(多为 `stereotyping`/`slur`),而 summary 本身仍然忠实;仅 1 例 (`7w5xCkEiO6A`) 是内容级虚构(把破碎歌词断言为 slur 并推断敌意互动)。ZH Normal 层最重(6/10),呈系统性模式:**涉性医疗/娱乐内容 → spurious mechanism**。含义:用档案做检索键时低危害,但 mechanism 字段单独拿来做规则/切片会引入 benign 误伤。
2. **洗白集中在"毒性只在标题、画面 benign"的样本。** 3 例 whitewash (`XScP1AiMkNM` 牛油果酱、`TRFp4a4lD0o` Floozy 空乘、`BV1ru411E7Ph` 破鞋) 全部是标题携带毒性、模型只如实描述画面的模式——档案生成对标题文本的注意不足。
3. **W2 两例疑似标签噪声的复核结论并不对称。** `XScP1AiMkNM`:标题本身厌女+地域贬损,"标签噪声"假设被部分挑战,更像**档案洗白**(漏读标题);`QvPp8Q7QhWE`:标题指向 findom 恋物内容,标签有可能成立,但文本证据不足,判 insufficient-evidence,需看视频终审。
4. **发现 1 例反方向分歧:`BV1MU4y1D7Ks`(ZH test, gt=Normal)** 转写明确贬低无性恋群体、档案正确报出 targets=LGBTQ+/dehumanization——疑似 gt 漏标,建议人工复核该 test 样本。
5. **target 字段:填了的基本有依据 (6/7),但召回率低。** 60 条中仅 7 条非空;约 13 条明确攻击某群体的样本 target 为空(EN 如反同布道 `l3eUapefQog`,ZH 全部 '娘炮/泼妇/母夜叉' 类均未填 target)。ZH 尤其严重(30 条中仅 1 条非空)——这直接解释了记忆编辑实验中 ZH 按 target 字段切片的退化(见 DEMO_memory_editing.md)。
6. **模态错挂常见但低危害**(标题内容被写进 speech cue / "transcript includes"):EN 2 例、ZH 4 例;词面在标题存在,故不计幻觉,但提示 cue 字段的模态归属不可尽信。

## 逐条审计

### MHC (English)

#### `7_-y8KAjfWs`  — Hateful / harmful / split=train

- **判定: `faithful`** — 反 trans 内容,targets=transgender、stereotyping/mockery/explicit 全部有清晰转写依据。
- target 有依据: yes | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: Blatant Misogyny and entitlement with trans women!!
- **Transcript**: Pay close attention. Notice a trend here Day 12 of being a girl. we're going to go buy taampons. This thing does not go up inside you. I did a brand deal for taamp. I'm sorry. We have the design up expensive.You're telling me if someone says would you like this free paper bag that came with my groceries or would you like this $35000 bag Well just man, because trans girls don't want to put trash like you in our designer bag I we probably most likely a better candidate.😊あ.Dating then like you know your average fish day 145 of being a girl and I have been accused of causing a national tmpon supply shortage would you wantampons I'm just Notice we never see trans men, biological women who identifies men doing shenanigans like this.I guess that male misogyny transcends even after you now identify as a female.
- **档案**: targets=["transgender"] | mechanism=["stereotyping", "mockery"] | explicitness=explicit
- **modality_cues**: {"speech": "The speaker uses derogatory terms and makes mocking statements about trans women's experiences with tampons."}
- **neutral_summary**: The video features a speaker making derogatory comments about trans women's experiences with tampons, using mocking and stereotypical language.

#### `Mqq5JiYQI04`  — Hateful / harmful / split=train

- **判定: `insufficient-evidence`** — 转写仅音乐;档案的 on_screen_text(cuckold 定义)属 OCR 断言,文本侧无法证实/证伪;Hateful 标签的依据(若有)不在可核文本内,无法裁决。
- target 有依据: n.a. | mechanism 相符: n.a. | 幻觉: no | 洗白: no
- **Title**: Cuckold 👀
- **Transcript**: 🎼 .
- **档案**: targets=[] | mechanism=["coded_language"] | explicitness=explicit
- **modality_cues**: {"on_screen_text": "Cuckolding is the act in which a man or woman enjoys watching their partner having sex with another person."}
- **neutral_summary**: The video includes a definition of cuckolding, which is presented as a term used to describe a specific sexual activity involving three people. The visual and textual content does not depict any individuals or scenarios but instead provides a straightforward explanation of the term.

#### `Ms1jmbrLbmU`  — Hateful / harmful / split=train

- **判定: `faithful`** — 'Chelsea rent boys' 恐同辱骂口号,标题+speech cue 一致;mechanism=slur 正确。遗漏:target 字段空(应为 gay men),记 target 召回缺失。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: 31/10/2015 - Liverpool fans at Chelsea singing: "Chelsea rent boys".
- **Transcript**: ...
- **档案**: targets=[] | mechanism=["slur"] | explicitness=explicit
- **modality_cues**: {"speech": "Chelsea rent boys"}
- **neutral_summary**: The video shows a crowd of people, including police officers, gathered outdoors. A person in the crowd is heard saying 'Chelsea rent boys'.

#### `TBPLkZzJdmI`  — Hateful / harmful / split=train

- **判定: `faithful`** — targets=homosexuals 有明确转写依据;slur/stereotyping 成立;threat 的解读略激进(转写 'I got a gun here' 语境混乱,疑似在描述他人),但有字面依据。
- target 有依据: yes | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: Islam and Homosexuality
- **Transcript**: 🎼Just like Christian communities in the Bible Bel can say ridiculous things. You don't going to hell, you know, shut up.😡Be quiet, Redneck, Donald Trump and these guys. be quiet. and they're allowed to do that anyhow. And I got a gun here. and they're walking around with guns and threatening behavior. And they're okay. But as Muslims, we have to be victimized. Otherwise, objectjected for for homosexual into a note, what we believe firmly wholeheartedly, definitely, blatantly, obviously, willingly, that homosexuality, if done in practice is simpleful, and we will never back down.
- **档案**: targets=["homosexuals"] | mechanism=["slur", "stereotyping", "threat"] | explicitness=explicit
- **modality_cues**: {"speech": "The speaker uses derogatory terms and makes threats."}
- **neutral_summary**: A speaker is addressing an audience while holding a microphone. They make derogatory comments about homosexuality and threaten violence against those who hold different beliefs.

#### `TRFp4a4lD0o`  — Hateful / harmful / split=train

- **判定: `whitewash`** — 标题 'Floozy Flight Attendants' 用贬损词攻击(女性)空乘,转写只有音乐;档案 targets/mechanism 全空、explicitness=none、summary 纯 benign 描述画面——标题毒性被完全洗掉。
- target 有依据: n.a. | mechanism 相符: n.a. | 幻觉: no | 洗白: yes
- **Title**: Floozy Flight Attendants Welcome You Inside The Plane - Emirates
- **Transcript**: 🎼
- **档案**: targets=[] | mechanism=[] | explicitness=none
- **modality_cues**: {"visual": "The video shows flight attendants in uniform walking down the aisle of an airplane, interacting with passengers."}
- **neutral_summary**: The video depicts flight attendants in uniform walking through the aisle of an airplane, engaging with passengers. The setting appears to be inside an aircraft, likely during a pre-flight or post-flight service.

#### `XScP1AiMkNM`  — Hateful / harmful / split=train **[W2 疑似标签噪声,强制纳入]**

- **判定: `whitewash`** — 标题本身含贬损内容(把他国女性称为 odalisque & concubine,厌女+地域攻击),但档案只描述画面(牛油果酱)且 explicitness=none、summary 全然 benign——档案忽略了标题文本的毒性。同时这也部分挑战 W2 的'纯标签噪声'假设:hateful 信号在标题而非画面。mechanism=coded_language 与内容对不上。终审需看视频确认画面确无其他内容。
- target 有依据: n.a. | mechanism 相符: no | 幻觉: no | 洗白: yes
- **Title**: Guyana produce wifey - your country produce odalisque & concubine.
- **Transcript**: (空转写)
- **档案**: targets=[] | mechanism=["coded_language"] | explicitness=none
- **neutral_summary**: The video shows a close-up of homemade guacamole in a bowl with a spoon, accompanied by the text 'Homemade guacamole' and 'Yay'.

#### `YGMvfNIIL7A`  — Hateful / harmful / split=train

- **判定: `faithful`** — male prostitutes 目标与 slur/stereotyping 均有充分转写依据。遗漏了 'smaller for our Asian members' 的种族歧视笑话。
- target 有依据: yes | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: MANwhore union PT2
- **Transcript**: But I know Heinez would like to be remembered for most was being a male prostitute. He was also a supercilious countnt.Jssega, G M Carlow, Silver back He bit. hiinz Hammer was undercu all of us. Here. He was charging the same price for stray eggs as he was for a Turkish snow cone.Oh, yes, he was also charging the same price for a Belgian steamer as he was for a Portuguese breakfast as we sell. So shall we reap.😊Thank you, Jean Carfe those kind words. Until this insane killer is apprehended, the union is going to institute a new security measure. This ring locked snugly over your vagina.And then communicates directly with a global satellite system, the rings of all sizes available, smaller for our Asian members.
- **档案**: targets=["male prostitutes"] | mechanism=["slur", "stereotyping"] | explicitness=explicit
- **modality_cues**: {"speech": "The video includes derogatory terms and stereotypes about male prostitutes."}
- **neutral_summary**: The video contains derogatory comments about male prostitutes, using slurs and stereotypical language.

#### `YmCLjVMi8Kw`  — Hateful / harmful / split=train

- **判定: `faithful`** — targets=drag queens、implicit stereotyping 与转写(反 drag queen 进教堂的质问)一致。
- target 有依据: yes | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: DRAG QUEENS IN CHURCHES! 🤦‍♂️ #christianity #christian #church #lgbt #islam #god #liberal #dragqueen
- **Transcript**: I have a question for you, Christians. See, Ive read the Bible thrown and through, but I didn't see any verse that says that drag queens are to be hosted in a church to give sermons while the pastor is sitting there listening to her. Have you guys become this liberalized.
- **档案**: targets=["drag queens"] | mechanism=["stereotyping"] | explicitness=implicit
- **modality_cues**: {"speech": "The speaker questions the presence of drag queens in churches, suggesting they haven't found any biblical verses supporting this practice."}
- **neutral_summary**: The video discusses the presence of drag queens in churches, questioning whether there are biblical verses supporting their inclusion. The speaker appears to be addressing Christian viewers.

#### `cXRgVEENkPA`  — Hateful / harmful / split=test

- **判定: `faithful`** — religious group + sexual orientation 两个 target 均有转写依据;stereotyping/mockery 成立。
- target 有依据: yes | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: Catholic Drag Queens
- **Transcript**: The Los Angeles Dodgers found themselves in hot water this week when they invited the sisteristers of perpetual indulgence to participate in the team's upcoming gay pridede event that you may be asking yourself, wait who are these sisters and what exactly are they indulging in Well apparently this is a group of male drag queens who dress up as catho nuns to mock Christianity, the invitation angered many Catholichos understandably which led the Dodgers to disinvite the sisters and that in turn offended the LGBtq.I a plus community, I think I got them all and with any dispute between Christianity and unorthodox sexuality, we know which wins and right on cue, the Dodgers issued an apology and reinvited the Ss to gay pridede Knight in hopes they would accept a community hero award.
- **档案**: targets=["religious group", "sexual orientation"] | mechanism=["stereotyping", "mockery"] | explicitness=explicit
- **modality_cues**: {"visual": "A person dressed as a nun is seen, and there is a Dodgers logo with a baseball and sun symbol.", "speech": "The transcript discusses Catholic drag queens, their participation in a gay pride event, and the conflict between Christianity and LGBTQ+ communities."}
- **neutral_summary**: The video discusses the controversy surrounding the Los Angeles Dodgers' invitation of Catholic drag queens to a gay pride event, highlighting the tension between religious groups and LGBTQ+ communities.

#### `xqgjkn3uh0w`  — Hateful / harmful / split=test

- **判定: `faithful`** — stereotyping/implicit 有依据;但 summary 把攻击对象错框为'对 GTA 玩家的刻板印象'并用 humorously 软化,实际贬损对象是性工作者;target 字段空,记召回缺失+轻度软化。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: POV: GTA Players when they see a hooker #shorts #gta #gaming
- **Transcript**: do. they always do that shit. Oh, we have here.🎼What on. Ain't got no money. Fuck me. I got no. I gotta hell of money. I got put something out of my bank. spin. I' must been likehu. I'm gonna take a K, K. know some light, some light. You know what saying All, I get your bit here.We need some privacy, baby.I bet don't say less than I'm,' my privacy is my middle name had us 12 seconds later. What fuck' trying to blow me up, but I didn't do shit to you All, What fuck's trying to blow me up, bro.I I got I know I got us, but like you know, oh my god,, dumb dumb, get look at you, she's still in here, oh she is still in here.D这来就是专麦怎么。😊
- **档案**: targets=[] | mechanism=["stereotyping"] | explicitness=implicit
- **modality_cues**: {"visual": "The video shows a nighttime cityscape with a car driving through it.", "on_screen_text": "GTA Players when they see a hooker"}
- **neutral_summary**: The video depicts a nighttime scene in a city, showing a car driving through the streets. The on-screen text humorously suggests a stereotype about Grand Theft Auto players reacting to seeing a prostitute.

#### `-cgm2EZcLC8`  — Offensive / harmful / split=test

- **判定: `faithful`** — speech cue 与转写逐字一致;粗口辱骂标 slur/explicit 合理(严格说是 insult 而非 slur)。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: Kensington Love & Drugs
- **Transcript**: I will, because you can't hold me after a pussy you pussy.🎼.
- **档案**: targets=[] | mechanism=["slur"] | explicitness=explicit
- **modality_cues**: {"speech": "Kensington Love & Drugs . I will, because you can't hold me after a pussy you pussy."}
- **neutral_summary**: The video appears to show a street scene with individuals walking and standing on a sidewalk. There is a visible trash can and some litter on the ground. The audio includes a phrase that seems to be a slur.

#### `C1NYlIk5TwQ`  — Offensive / harmful / split=train

- **判定: `faithful`** — 转写严重破碎,cue 基本是标题复述;stereotyping 依据弱;explicitness=none 与 Offensive 标签的张力无法从文本裁决。
- target 有依据: n.a. | mechanism 相符: no | 幻觉: no | 洗白: no
- **Title**: Sex workers in Nairobi complaining on how low  price they ofger
- **Transcript**: A么 w.And last time only, Missouri around and name when you.Iita call me and be in for man..
- **档案**: targets=[] | mechanism=["stereotyping"] | explicitness=none
- **modality_cues**: {"speech": "Sex workers in Nairobi complaining on how low price they offer."}
- **neutral_summary**: The video features a person speaking about sex workers in Nairobi, mentioning low prices and a previous encounter.

#### `DTcm4d7S4Fs`  — Offensive / harmful / split=train

- **判定: `faithful`** — summary 忠实(无神论者论证'上帝纵容强奸');mechanism=stereotyping 依据弱(更接近亵渎/挑衅内容,非群体刻板印象),记字段弱依据。
- target 有依据: n.a. | mechanism 相符: no | 幻觉: no | 洗白: no
- **Title**: Actually, God Is COMPLETELY Fine With Rape #atheism #christianity #atheist
- **Transcript**: So there are a lot of secular reasons as to why grape is wrong, but I would also like to point out how there are a lot of instances in the Old Testament where God seemingly approves of grape in one way or another, the entire system where the man pays for a woman is gring the woman like in the the sexual aspect of being in a marriage or marrying somebody, if you're pain,For that person and not getting their consent then that is gring them it's a form of sexual slavery God obviously doesn't have a problem with grape, He doesn't have a problem prescribing grape in certain instances, in order for his people to propagate and multiply, as he said in the Bible,
- **档案**: targets=[] | mechanism=["stereotyping"] | explicitness=explicit
- **modality_cues**: {"speech": "The speaker discusses the Old Testament and God's apparent approval of rape."}
- **neutral_summary**: A person wearing a shirt with the word 'Heathen' speaks about the Old Testament and God's apparent approval of rape, using religious terminology.

#### `IFP0l5EUO70`  — Offensive / harmful / split=test

- **判定: `faithful`** — 'Minor Attracted Persons' 在标题而非转写,speech cue 属模态错挂;summary 的 'p3dos' 屏幕文字属 OCR 断言不可证;内容要旨(嘲讽 MAP 委婉语)与标题相符。记 1 例模态错挂。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: Minor Attracted Persons? #airsoft #based
- **Transcript**: 🎼I get messages from the store when you're making love to me.
- **档案**: targets=[] | mechanism=["slur"] | explicitness=explicit
- **modality_cues**: {"speech": "The speaker uses the term 'Minor Attracted Persons' as a slur."}
- **neutral_summary**: A person wearing protective gear is holding a firearm outdoors. The video includes on-screen text that provides a humorous or sarcastic reaction to a previous statement about a group being called 'p3dos' instead of 'Minor Attracted Persons'.

#### `Jocr_4Py5-U`  — Offensive / harmful / split=test

- **判定: `faithful`** — insult 有依据(scumbag);'slur' 略过标;on_screen_text 断言不可证但与内容不矛盾。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: Your Scumbag Dad - The Milk Run!
- **Transcript**: Hey, it's me your scumbag dad dad's going to go on a little trip to get some milk, I need you to tell me where we put those fun pretend passports that we made together be right back.Okay, God's leaving I can't believe youy how many women have been talking to behind my back look I'm not going lie you, I lost cow No you can't go this room's not.🎼都贝.Oh my God. hey, relax. I definitely wasn't trying to run away. I was just getting the milk. This is the Maldves. It's famous for its milk. So so dad did a good job. I brought the milk. Dad just got broken up with. so he's gonna finish this champagne for two. I'll catch up with you later, okay.Hey you want to go on the swing, here we go, one, two, here's the big one, three,, your leg is not supposed to bend that way.
- **档案**: targets=[] | mechanism=["slur", "insult"] | explicitness=explicit
- **modality_cues**: {"visual": "The person is wearing a straw hat and a colorful shirt, suggesting a casual setting.", "speech": "The speaker uses derogatory terms like 'scumbag' and 'cow,' which are insults.", "on_screen_text": "The text overlay includes phrases like 'Your Scumbag Dad!' and 'I'm going home.'"}
- **neutral_summary**: The video features a person speaking in a casual indoor setting, using insulting language towards someone referred to as 'Scumbag Dad.' The scene then transitions to an outdoor location where the same person is seen swinging on a swing, with a caption suggesting they are in the Maldives.

#### `QvPp8Q7QhWE`  — Offensive / harmful / split=train **[W2 疑似标签噪声,强制纳入]**

- **判定: `insufficient-evidence`** — W2 疑似标签噪声样本。标题 'Paypig boyfriend & mistress' 指向 findom 恋物内容,Offensive 标签有可能是对的;档案却断言'no indication of any harmful or offensive content',至少忽略了标题线索,是轻度洗白倾向;但文本毒性不'明显',按规则记 insufficient-evidence,终审需看视频。
- target 有依据: n.a. | mechanism 相符: n.a. | 幻觉: no | 洗白: partial
- **Title**: Paypig boyfriend & mistress Girlfriend
- **Transcript**: 20,40, 60,8100121461 82,2224 to 60.Say thank you, Thank you, Mr. Y.
- **档案**: targets=[] | mechanism=[] | explicitness=none
- **modality_cues**: {"visual": "A person is counting money while another person holds their leg."}
- **neutral_summary**: The video shows a person counting money while another person holds their leg. There is no indication of any harmful or offensive content.

#### `Y05AKSPwElU`  — Offensive / harmful / split=train

- **判定: `faithful`** — 贬损与性化言语在转写中可证('toxic'、'bit(ch)');cue 中 'Latina girl' 来自标题。target 字段空(受害对象为具体个人,可接受)。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: Rapper X4 shuts down Latina girl during the middle of the interview
- **Transcript**: I can't have this conversation on this camera they just they told you that they're not on touch it already.Okay, you said it four times, I'm recording Google Expo and Google, Googley, Google Expo and Google the street TV, dis to Hto.This to who, we can be record right here. All right dont don't take it to the next level, Just go inside your house. Just go inside the house. We just told you're not going touch it. but if you take it to the next level, we' gonna do more than touch it. So just go on the house, please, you like that. you toxic it come.You talk me. What's your Instagram. She starts smilinging when I said that you look kind of good too like you.She toxic. toxic. I know what toxic when I see. He yeah,s got that good. Yeah, yeah, yeah, Yeah, you see her. she liked that. she smiling. She she like that shit.'t you like it that shit you get a certain way. like it that shit We pitch on the deal. I'm telling you be surprised What type of bit like it.😊
- **档案**: targets=[] | mechanism=["slur", "insult"] | explicitness=explicit
- **modality_cues**: {"speech": "The rapper uses derogatory terms towards a Latina girl and insults her."}
- **neutral_summary**: A rapper is seen speaking to the camera, using derogatory terms and insults towards a Latina girl during an interview. The video includes visual and audio elements of the rapper's interaction.

#### `k9OtaMbK0Ac`  — Offensive / harmful / split=test

- **判定: `faithful`** — speech cue 与转写一致,insult/explicit 有依据(脏话+斥责)。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: helicopter 🍆 in front of my parents #shorts
- **Transcript**: Helicopter D in front of my parents.Gd, stop fucking doing that. it is pretty weird dude. And if he thinks it is, it's fucking weird.😡
- **档案**: targets=[] | mechanism=["insult"] | explicitness=explicit
- **modality_cues**: {"speech": "The person speaking says 'Helicopter 🍆 in front of my parents.'"}
- **neutral_summary**: A person is standing in a living room, wearing a towel around their waist, while others are sitting on a couch. The person speaking mentions a helicopter in front of their parents, using an insult.

#### `l3eUapefQog`  — Offensive / harmful / split=train

- **判定: `faithful`** — summary/cue 忠实转述反同布道内容;但 target 字段空是明显召回缺失——homosexuals 是全文唯一且明确的攻击目标(转写把同性恋称为 abomination)。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: Voddie Baucham \| What's wrong with Homosexuality? #lgbt #pride #gospel #voddiebaucham #bible
- **Transcript**: What's wrong with homosexuality, A number of things, but just a few in this context. numberer one, it's a violation of the created order. It's a violation of the created order. It's not how we remain. Secondly, it denounces procreation categorically. Thirdly, it blasphemes the illustration. And this is especially true when we understand the illustration of Christ and his bride the church. And then finally, it denies the very need for sanctification because it takes what God calls sinful and calls it righteous.God calls this an abomination and we instead call it righteous. It's the only sin, by the way, for which God destroyed cities with fire in brimstone. It's unique. It's not like other sins, It's unique, not all sins are called abominations, homosexuality is unique in that regard, very few sins in that category, and not all sins whether ended in God destroying twin cities with fire and brimstone. It's unique in that regard.
- **档案**: targets=[] | mechanism=["stereotyping"] | explicitness=explicit
- **modality_cues**: {"speech": "The speaker discusses the perceived violations of the created order, denial of procreation, and the unique nature of homosexuality as an abomination."}
- **neutral_summary**: The video features a speaker discussing the theological implications of homosexuality, emphasizing its perceived violation of the created order, denial of procreation, and its unique status as an abomination.

#### `uFqdpifgHO0`  — Offensive / harmful / split=train

- **判定: `faithful`** — 转写为空,'Man whore' 在标题;speech cue 模态错挂,但 summary 自己写明'video title includes the phrase',内容判断正确。记 1 例模态错挂。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: Man whore
- **Transcript**: (空转写)
- **档案**: targets=[] | mechanism=["slur"] | explicitness=explicit
- **modality_cues**: {"speech": "Man whore"}
- **neutral_summary**: The video shows a person sitting indoors, possibly in a bedroom, with a casual posture. The video title includes the phrase 'Man whore,' which appears to be a derogatory term directed at a man.

#### `6nyI-ZflSRU`  — Normal / normal / split=val

- **判定: `faithful`** — benign 描述一致。
- target 有依据: n.a. | mechanism 相符: n.a. | 幻觉: no | 洗白: no
- **Title**: #hustle #hustler #grind #rip #youngdolph
- **Transcript**: I just want everybody to know it like, but you don't need nobody or their validation or their opinion or know nothing, knowing saying like whatever you think.
- **档案**: targets=[] | mechanism=[] | explicitness=none
- **modality_cues**: {"visual": "A hand is seen holding a stack of US dollar bills."}
- **neutral_summary**: The video shows a close-up of a hand holding a bundle of US dollar bills.

#### `7w5xCkEiO6A`  — Normal / normal / split=train

- **判定: `hallucination`** — 内容级幻觉:档案断言转写含 slur 'Bonage three songss' 并推断'hostile interaction'——该串是破碎歌词,不是 slur;60 条中唯一的内容级虚构判读。
- target 有依据: n.a. | mechanism 相符: no | 幻觉: yes (content-level) | 洗白: no
- **Title**: Justin & Riley \| The Lake
- **Transcript**: I'm really sorry, please don't help Maie Wayne.I never did this amazing.🎼T quiet, you know so how.Insecurities acus myself.I'm bitter and shake the sweetest temptation check in, if you call.Bonage three songss, I think you're going to need to update your profile there, timber snakeake.
- **档案**: targets=[] | mechanism=["slur"] | explicitness=explicit
- **modality_cues**: {"speech": "The video contains a slur 'Bonage three songss' which is used as part of the dialogue."}
- **neutral_summary**: The video features two individuals in close proximity, with one person speaking. The dialogue includes a slur, suggesting a hostile interaction. The setting appears to be outdoors, possibly near a lake.

#### `9op067bRp9A`  — Normal / normal / split=train

- **判定: `faithful`** — Normal 样本,benign 描述与转写一致;WLW 内容未被过度打标(与部分样本的过火形成对照)。
- target 有依据: n.a. | mechanism 相符: n.a. | 幻觉: no | 洗白: no
- **Title**: 🏳️‍🌈 WLW(lgbtq+)🌈 Tiktok by caitlinandleahh
- **Transcript**: 🎼Cudddle positions on our Emma mattress The little spoon, the starfish, the listen to heartbeat.The caddling our baby the touch be it to hot but still want to show affection.
- **档案**: targets=[] | mechanism=[] | explicitness=none
- **modality_cues**: {"visual": "Two individuals are lying on a mattress in various cuddle positions.", "on_screen_text": "The text 'The starfish', 'The listen to heart beat', 'The it's too hot but still want to show affection' appears."}
- **neutral_summary**: The video showcases two individuals demonstrating different cuddle positions on an Emma mattress. Text overlays suggest playful and affectionate themes related to the positions.

#### `GgZBFhWbyiA`  — Normal / normal / split=train

- **判定: `faithful`** — 角雕鸟类视频,完全 benign,档案一致。
- target 有依据: n.a. | mechanism 相符: n.a. | 幻觉: no | 洗白: no
- **Title**: Harpy Eagle Bird #viral #nature #shorts #like
- **Transcript**: (空转写)
- **档案**: targets=[] | mechanism=[] | explicitness=none
- **modality_cues**: {"visual": "A Harpy Eagle is perched on a tree branch, surrounded by foliage, with a white bird nearby."}
- **neutral_summary**: The video shows a Harpy Eagle perched on a tree branch, surrounded by greenery, with a white bird nearby. The scene appears to be set in a natural environment.

#### `TZRqW_SDxW4`  — Normal / normal / split=train

- **判定: `hallucination`** — 亲 LGBT 的科普/玩梗内容(历史名人性向),summary 忠实;但 targets=['gender'] 无依据、stereotyping+explicit 过火——本样本是全部 60 条中唯一'target 字段无依据'的实例。也提示:LGBT 话题的 benign 内容会被字段误伤(与记忆编辑实验的切片语义相关)。
- target 有依据: no | mechanism 相符: no | 幻觉: yes (field-level) | 洗白: no
- **Title**: 3 LEGENDS YOU DIDN'T KNOW WERE GAY or BI 🏳️‍🌈 #lgbt #history #gay #bi
- **Transcript**: Three legendary people. you didn't know We're gay. The last one had the hottest boyfriend. Shakespeare is an entire Wikipedia on his sexuality because he wrote so many love poems for men, and one of them he calls a man, the master mistress of my passionsh. Leonardo da Vinci was gay, lefthanded, vegetarian and easily distracted. when he was 23 years old, the Italian government accused him of sodomy. if he were convicted. The punishment would have been the death penalty. Julia Caesar An Rome was the gayest, and Caesar had a very public affair with this very Hawkeye. Nikoid's king of Bethhenia.😊
- **档案**: targets=["gender"] | mechanism=["stereotyping"] | explicitness=explicit
- **modality_cues**: {"speech": "The speaker discusses historical figures who were allegedly gay, using terms like 'hot boyfriend' and 'master mistress'.", "on_screen_text": "#3 CAESAR (Wait for his BF 😊)"}
- **neutral_summary**: The video discusses historical figures, specifically mentioning Julius Caesar and Leonardo da Vinci, and suggests they were gay. It uses playful language and includes a reference to Caesar having a 'very public affair.'

#### `Tc20odJbbOc`  — Normal / normal / split=train

- **判定: `faithful`** — street quiz 关于 Grindr 声音,implicit stereotyping 有依据(straight/gay 预设),summary 准确。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: Can straight guys name **this** gay sound 👀🫢
- **Transcript**: Claire, are you straight for $20. Can you name this sound, please., that sounds super familiar. God, I can't give it to you. Okay, no worry. Thank you so much for trying though, man. It was grinder sound. Oh my heard for that bro Oh my god, You had an ad for grinder. I heard Okay, A lot of people tell me about it. Well, I never tried it out. Are you guys straight. Yeah, straight, So for $20. Can you name this sound..That's like a person losing another person. I don't know. I that Rob, are you straight. Yeah, I'm straight. Yeah, Rob, So for $20, can you name this sound.😊Oh my God, you know, I've heard this sound.I've heard this. don't tell my wife, I I haven't played Tinder a long time. that sounds like it's from Tinder. If it's Tinder Bo gay people. Oh grinder. Okay, here you go. You got it. sure. Yeah, I'm confident. Thanks so much. Thank you.😊
- **档案**: targets=[] | mechanism=["stereotyping"] | explicitness=implicit
- **modality_cues**: {"speech": "The conversation revolves around identifying a sound associated with a dating app, specifically mentioning 'Tinder' and 'grinder'."}
- **neutral_summary**: The video features a group of individuals engaging in a casual conversation on a city street at night. They discuss a sound they recognize as being related to a dating app, with one person suggesting it might be from 'Tinder'. The conversation appears friendly and light-hearted.

#### `VyuyhTiWimA`  — Normal / normal / split=train

- **判定: `faithful`** — 'handjob roulette' 性暗示内容,sexual_harassment/explicit 有标题依据;相对 Normal 标签属过度打标,但非虚构。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: TikTok - handjob roulette
- **Transcript**: Oh, fuck again.
- **档案**: targets=[] | mechanism=["sexual_harassment"] | explicitness=explicit
- **modality_cues**: {"visual": "A hand is seen holding a bottle of alcohol, and a bottle cap is shown rolling on a table.", "on_screen_text": "Whos ready for some handjob roulette"}
- **neutral_summary**: The video shows a hand holding a bottle of alcohol and a bottle cap rolling on a table. The on-screen text reads 'Whos ready for some handjob roulette'.

#### `WETQcPxy7y0`  — Normal / normal / split=test

- **判定: `hallucination`** — 包皮环切医疗广告,summary 本身忠实;但 mechanism=['stereotyping'] 无任何依据——字段级虚报(低危害:explicitness=none,summary benign)。
- target 有依据: n.a. | mechanism 相符: no | 幻觉: yes (field-level) | 洗白: no
- **Title**: Penis का खतना करवाना चाहिए  \| Stapler Circumcision \|  Bloodless, Painless & Fast Recovery
- **Transcript**: (空转写)
- **档案**: targets=[] | mechanism=["stereotyping"] | explicitness=none
- **modality_cues**: {"visual": "The video shows a close-up of a medical instrument being used on a model of a penis.", "on_screen_text": "The on-screen text includes the hospital's name, contact number, and address."}
- **neutral_summary**: The video appears to be an advertisement for a medical procedure related to circumcision, featuring a close-up of a medical instrument being used on a model of a penis. The on-screen text provides contact information for a hospital.

#### `ZOIF6HGRu0A`  — Normal / normal / split=train

- **判定: `faithful`** — 受害者自述被性侵经历,summary 忠实转述;mechanism=sexual_harassment 描述的是被转述的事件而非视频对他人的攻击——暴露 mechanism 字段'话题 vs 攻击'的语义歧义,建议 schema 澄清。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: My Sister SEXUALLY Abused me!!! 😢😭Reddit #shorts #askredditpeople
- **Transcript**: My sister sexually abused me. My parents are mentally abused with a whole lot of drugs. While my older sister sexually abused me when I was eight years old, I had learnt sex ed at school or how anything with the human body works.One day, my older sister, she's 14 at the time, coerccing me to lay down on my bed. Then her asking me to take off my trousers and have her lay on top of me with her trousers and underwear off as well.I was too young to know any better. And I never told my family or friends at school. It continued on for several months, about twice a week for 10 or so minutes. My sister said it's cool. So I guess that was my rationale for going through with the whole thing.😊I never thought I'd get to say this, but I can't wait to leave my fucked up family behind where I don't really have to see them again. because in their actions, they have fucked me and my life up much more than I'll ever know. I'm my own man now and intend on keeping it that way.And when I have kids, I'm going to give them the best fucking family life ever, because I don't want them to have to go through what I have.
- **档案**: targets=[] | mechanism=["sexual_harassment"] | explicitness=explicit
- **modality_cues**: {"speech": "The speaker describes being sexually abused by their older sister."}
- **neutral_summary**: The video discusses the speaker's experience of being sexually abused by their older sister, who was 14 at the time. The speaker mentions learning about sex education at school and the abuse continuing for several months. The speaker expresses a desire to leave their family behind and promises to provide a better family life for their future children.

#### `fGng_nrDB0o`  — Normal / normal / split=train

- **判定: `faithful`** — mean-girl POV 短剧,insult/implicit 依据弱但可接受(阴阳怪气台词)。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: #pov the mean girl makes you uncomfortable during ur presentation…#shorts
- **Transcript**: I was never scared Angela, let's save questions until the end of Jane's presentation Yeah sorry I'm just like confused.
- **档案**: targets=[] | mechanism=["insult"] | explicitness=implicit
- **modality_cues**: {"speech": "The speaker expresses confusion and discomfort during a presentation."}
- **neutral_summary**: A person is speaking about feeling confused and uncomfortable during a presentation. The video includes a transcript with a comment suggesting saving questions until the end of the presentation.

### MHC_zh (Chinese)

#### `BV12V411N7ti`  — Hateful / harmful / split=train

- **判定: `faithful`** — 对岸田文雄的政治辱骂,insult/explicit 正确;OCR 断言与标题一致;攻击对象为个人,target 空可接受。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: 岸田文雄大<em class="keyword">傻逼</em>
- **Transcript**: 🎼.。my sex with my.
- **档案**: targets=[] | mechanism=["insult"] | explicitness=explicit
- **modality_cues**: {"on_screen_text": "岸田文雄领着排核污染水，大傻逼!"}
- **neutral_summary**: The video features a black-and-white image of a person at a podium with microphones, likely during a speech or press conference. The on-screen text includes derogatory language directed at a political figure.

#### `BV1DA411y7vy`  — Hateful / harmful / split=train

- **判定: `faithful`** — summary 称'transcript includes 傻逼'但转写为空——词在标题,属模态错挂而非内容虚构;slur/explicit 判定本身正确(对戈培尔的辱骂)。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: <em class="keyword">傻逼</em>戈培尔
- **Transcript**: (空转写)
- **档案**: targets=[] | mechanism=["slur"] | explicitness=explicit
- **modality_cues**: {"speech": "<em class=\"keyword\">傻逼</em>戈培尔"}
- **neutral_summary**: The video features a black-and-white portrait of a man dressed in formal attire. The transcript includes the word <em class="keyword">傻逼</em>, which is a derogatory term in Chinese.

#### `BV1LW4y1J7eY`  — Hateful / harmful / split=train

- **判定: `faithful`** — 标题'傻逼父母带的傻逼小孩'被挂到 speech cue(转写是行车对话),记模态错挂;summary 判定 offensive language directed at parents and children 有标题依据。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: <em class="keyword">傻逼</em>父母带的<em class="keyword">傻逼</em>小孩
- **Transcript**: 你也要啊哦，你为什么知道你不要呢？你可以自己找啊，你怎么知道你找得到呢？你我有没带小朋友的。😡
- **档案**: targets=[] | mechanism=["slur"] | explicitness=explicit
- **modality_cues**: {"speech": "<em class=\"keyword\">傻逼</em>父母带的<em class=\"keyword\">傻逼</em>小孩"}
- **neutral_summary**: The video shows a car driving through a city street at dusk, with traffic lights and other vehicles visible. The transcript contains offensive language directed at parents and children.

#### `BV1NZ421r7eh`  — Hateful / harmful / split=test

- **判定: `faithful`** — cue 引用属实,insult 成立;但 summary 说攻击'unspecified person'——标题 '傻逼小日子' 的攻击目标是日本人,target 召回缺失。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: 波兰球<em class="keyword">傻逼</em>小日子沙雕日常
- **Transcript**: 🎼你别跳，我不想活了，你想象你媳妇儿，媳妇儿跟人跑了，你还有兄弟，就是跟我兄弟跑的，你想想你家孩子还是他俩的死吧。妈，你活的也没啥价值。😡皮皮虾皮着一下很开心。😊
- **档案**: targets=[] | mechanism=["insult"] | explicitness=explicit
- **modality_cues**: {"visual": "The video features a cartoon building with various flags and emojis surrounding it.", "speech": "The transcript includes phrases like '傻逼' (idiomatically insulting term), '你别跳' (don't jump), and '你活的也没啥价值' (you don't have much value)."}
- **neutral_summary**: The video depicts a cartoon building with various flags and emojis, accompanied by a transcript containing insulting and devaluing language towards an unspecified person.

#### `BV1VG411o7NR`  — Hateful / harmful / split=train

- **判定: `faithful`** — 转写反复出现'傻逼',slur/explicit 成立;粉色女性符号(视觉断言,不可证)若属实则 Hateful 标签可能指向厌女梗,档案未点破,记潜在 target 召回缺失。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: ⚡I'm a <em class="keyword">傻逼</em>⚡
- **Transcript**: 🎼。😊傻逼咋逼咋咋咋咋咋傻逼。😡
- **档案**: targets=[] | mechanism=["slur"] | explicitness=explicit
- **modality_cues**: {"speech": "I'm a <em class=\"keyword\">傻逼</em>"}
- **neutral_summary**: The video features a person with a coconut head and text that includes a slur. The visual elements include a person with a coconut-like head and a pink female symbol.

#### `BV1bk4y1K7Eu`  — Hateful / harmful / split=train

- **判定: `faithful`** — 日军暴行标题,dehumanization/explicit 与标题一致;OCR 断言与标题相符(细节'12-year-old'不可证但不矛盾)。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: 日军将小女孩<em class="keyword">下体</em>撕裂
- **Transcript**: 🎼.
- **档案**: targets=[] | mechanism=["dehumanization"] | explicitness=explicit
- **modality_cues**: {"visual": "The video depicts a historical scene with soldiers and a young girl, but the focus is on the overlaid text.", "on_screen_text": "The text describes a violent act against a 12-year-old girl, using explicit language."}
- **neutral_summary**: The video shows a historical scene with soldiers and a young girl, accompanied by overlaid text describing a violent act against her. The text uses explicit language to describe the incident.

#### `BV1he4y1N76x`  — Hateful / harmful / split=train

- **判定: `faithful`** — '妈宝男'嘲讽,stereotyping/implicit 有依据,cue 引用转写属实。target 空(可记'男性/妈宝男'召回缺失,弱)。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: 23岁，苏醒，但是是“妈宝男”
- **Transcript**: 🎼总是执着等待妈妈在哪里？在心里。别劝我分手。哎，苏妈妈是你好甜美。哎，苏妈妈你好嗯到对的人多么浪漫，天的妈妈有什么祝福要送给苏醒吗？啊，你可以可以啊，苏醒有话要跟你说，苏妈妈好的，好的嗯。😊🎼I think.
- **档案**: targets=[] | mechanism=["stereotyping"] | explicitness=implicit
- **modality_cues**: {"visual": "A person wearing a headset and a microphone is seen, suggesting they might be participating in a public event or performance.", "speech": "The transcript includes phrases like '执着等待妈妈在哪里？' which translates to '执着 waiting where mom is?' indicating a stereotypical depiction of a 'mom's boy'."}
- **neutral_summary**: The video features a person wearing a headset and microphone, possibly at a public event. The transcript contains a phrase suggesting a stereotypical portrayal of someone who is overly dependent on their mother.

#### `BV1iy4y1G7aD`  — Hateful / harmful / split=test

- **判定: `faithful`** — 转写全程'娘炮'对骂,slur+insult/explicit 正确,summary 还给出了 '娘炮' 指向女性化/同性恋男性的解释——语义到位,但 target 字段未填,记召回缺失。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: 双   响   炮
- **Transcript**: 🎼说谁他妈是娘炮呢？说你呢死娘炮操你有没有礼貌啊，你他妈说谁娘炮呢？我操。
- **档案**: targets=[] | mechanism=["slur", "insult"] | explicitness=explicit
- **modality_cues**: {"speech": "娘炮"}
- **neutral_summary**: The video shows a chat conversation where one person uses the term '娘炮' which is considered derogatory towards individuals perceived as effeminate or homosexual. The other person responds with insults and further derogatory comments.

#### `BV1pE411A787`  — Hateful / harmful / split=test

- **判定: `faithful`** — '娘炮' slur 判定正确;标题被挂到 speech cue(转写实际是'我操你看看真正的樱花'),记模态错挂;summary 称 slur 'directed at a specific group' 合理但未点名(女性化男性/LGBT)。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: 好吧，我承认我是<em class="keyword">娘炮</em>……
- **Transcript**: 🎼我操你看看真正的樱花。
- **档案**: targets=[] | mechanism=["slur"] | explicitness=explicit
- **modality_cues**: {"speech": "好吧，我承认我是<em class=\"keyword\">娘炮</em>……"}
- **neutral_summary**: The video contains a slur directed at a specific group, as indicated by the text '好吧，我承认我是<em class="keyword">娘炮</em>……' in the transcript.

#### `BV1wN411a7HT`  — Hateful / harmful / split=train

- **判定: `faithful`** — '娘炮/恶心/越来越娘' 在转写中可证;cue 点明 derogatory towards a certain gender identity——语义正确;target 字段仍为空,记召回缺失。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: 杨和苏:“怎么这么<em class="keyword">娘炮</em>啊？” 呱哥:“我真的无语=_=”
- **Transcript**: 你在干什么，怎么这么娘炮啊，还绿头发，真的太恶心了，我靠，越来越娘了啊。我哋黄啲。
- **档案**: targets=[] | mechanism=["slur", "insult"] | explicitness=explicit
- **modality_cues**: {"visual": "A person adjusts their face mask while others sit at desks in the background.", "speech": "The speaker uses the term '娘炮' which is derogatory towards a certain gender identity, and expresses disgust and frustration."}
- **neutral_summary**: The video shows a person adjusting their face mask while others are seated at desks in the background. The speaker uses a derogatory term and expresses strong negative emotions towards someone, likely based on gender identity.

#### `BV14C4y1G7Ey`  — Offensive / harmful / split=train

- **判定: `faithful`** — cue 称转写含'老巫婆'——该词在标题,日语转写中的对应物是'ババ(老太婆)/妖怪',要旨一致,记模态错挂(轻)。标题的性暗示('深入进去很舒服')未被档案反映。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: 老<em class="keyword">巫婆</em>舌头真长，深入进去很舒服很享受
- **Transcript**: 🎼うん、京都京頭はい、京頭先生が高速に厳しい方で目の前で廊下を走ったりすると、どこまでも追いかけられて。うこ捕また最後ブリ！🎼安声しなさいを 精神 いえ、あれは肉体的ダメージ相当！元陸上選手で、未だかつてあのキスから逃れられた者はいないという その二つなの 地上で最も大速に近いババ この格好いあんな妖怪だ！
- **档案**: targets=[] | mechanism=["slur"] | explicitness=explicit
- **modality_cues**: {"speech": "The transcript contains derogatory terms like '老<em class=\"keyword\">巫婆</em>' which translates to 'old witch'."}
- **neutral_summary**: The video features animated characters in a school setting, with one character expressing concern about being chased by someone running down the hallway. The transcript includes derogatory language directed at a specific individual.

#### `BV15W411C7FT`  — Offensive / harmful / split=test

- **判定: `faithful`** — stereotyping/implicit 有依据;但 summary 把'你多玩几年,你老婆也要被人玩多几年'的挑衅曲解为'担心妻子不忠',且标题'男人就要娶小老婆'的厌女前提未被点出——轻度软化+理解偏差。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: 男人就要娶<em class="keyword">小老婆</em>
- **Transcript**: 그.你是不是又换对象了？是的，你老这样玩不行的，为什么不想你没想过结婚吗？哥还年轻，结婚的事情不结，你还年轻啊，都三十好几了，三十好几怎么了？我还能玩多几年，那你想过没？你多玩几年，香音的你老婆也要被人玩多几年？😮😀老婆。
- **档案**: targets=[] | mechanism=["stereotyping"] | explicitness=implicit
- **modality_cues**: {"speech": "The video contains a conversation about marriage and age, with a focus on the speaker's own age and the potential for his wife to be unfaithful."}
- **neutral_summary**: The video features a dialogue about marriage and age, with the speaker expressing concerns about his wife's potential infidelity due to his own age.

#### `BV1CN4y1h79K`  — Offensive / harmful / split=train

- **判定: `faithful`** — 转写仅音乐;mechanism=slur 依据在标题'泼妇';summary 只描述画面(舞蹈+logo),未提标题毒性——字段与 summary 之间有张力但无虚构。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: 那个<em class="keyword">泼妇</em>真的不知道因果报应
- **Transcript**: 🎼
- **档案**: targets=[] | mechanism=["slur"] | explicitness=explicit
- **neutral_summary**: The video features animated characters in costumes performing a dance routine. The text overlay includes Chinese characters and the logo 'Bilibili'.

#### `BV1SH4y1D7kF`  — Offensive / harmful / split=train

- **判定: `faithful`** — 喜剧短剧,'奸夫淫妇'在转写中;summary 判断为 humorous/exaggerated 与内容相符;该 slur 本身未被 mechanism 点出(标了 stereotyping),记轻度欠标。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: 搞笑：大师说男女二人是奸夫<em class="keyword">淫妇</em>，男子生气质问，结局好搞笑
- **Transcript**: 奸夫淫妇大师，你说啥呢？一对狗男女过分了啊，老公你有事，我忘上。你啥意思啊，啥玩啥意思啊，他我媳妇儿管我叫老公呢。😡
- **档案**: targets=[] | mechanism=["stereotyping"] | explicitness=implicit
- **modality_cues**: {"visual": "A man is sitting on the ground while two other individuals walk past him.", "speech": "The text '过分了啊' appears, which translates to 'Too much.'", "on_screen_text": "The text '过分了啊' appears."}
- **neutral_summary**: The video depicts a scene where a man is sitting on the ground as two others walk by. There is text overlay that seems to express frustration or disapproval, but the context suggests a humorous or exaggerated scenario rather than a serious issue.

#### `BV1Tp4y1k7HG`  — Offensive / harmful / split=train

- **判定: `faithful`** — '妈宝男'吐槽,cue/summary 与转写高度一致,stereotyping/explicit 成立(对男性子群体的刻板化)。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: <em class="keyword">妈宝男</em>，到底有多可怕？
- **Transcript**: 🎼早上起来我们第一件事情不都是洗漱嘛，对吗？他起来第一件事摸手机给他妈打电话，妈我起床了，然后中午也得他妈打三四个电话，晚上睡觉之前还得给妈说晚安，妈妈我睡了，比如七八个电话挺正常的吧，跟朋友一起出门聚个会啊，逛个街什么的，他也得给他妈打个电话汇报一下，出去吃个饭，拍照片给他妈发过去，让他妈看我们今天吃什么，去逛个街买个衣服什么的，不得先问他妈他妈说哪家好看，就买哪家。前段时间我自己发的工资，我买个包，他他妈就说什么呃，这女孩子什么花钱。😊大手大脚啊，不节约，不勤俭持家。你说我花我自己的钱，关他妈什么事啊，我花我自己的钱，我自己劳动挣的钱，我我有错吗？对呀，你也知道我们还在谈恋爱，我们还没结婚呢，你妈就管那么宽，那迟早的事吗？谁要跟你过呀，你跟你妈过去吧，你这。😡
- **档案**: targets=[] | mechanism=["stereotyping"] | explicitness=explicit
- **modality_cues**: {"speech": "The speaker describes a '妈宝男' (a term used to describe a man who is overly dependent on his mother) and their behavior towards their mother."}
- **neutral_summary**: The video discusses a person's behavior towards their mother, using the term '妈宝男' which refers to someone who is overly dependent on their mother. The speaker criticizes this behavior and expresses frustration.

#### `BV1US4y1k7tq`  — Offensive / harmful / split=train

- **判定: `faithful`** — OCR=标题;summary 明确点出 '泼妇' 是 directed at women 的贬损语——机制与目标语义都对,但 target 字段未填,记召回缺失。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: 这才是真正的<em class="keyword">泼妇</em>，见者有粪！
- **Transcript**: 시.
- **档案**: targets=[] | mechanism=["slur"] | explicitness=explicit
- **modality_cues**: {"on_screen_text": "这才是真正的泼妇，见者有粪！"}
- **neutral_summary**: The video depicts a crowded street scene where individuals are walking and interacting. A text overlay in Chinese translates to 'This is the real泼妇, those who see will have feces!' indicating a derogatory slur directed at women.

#### `BV1gr4y1h7T1`  — Offensive / harmful / split=train

- **判定: `faithful`** — '母夜叉' 贬损称谓在转写中反复出现,cue 解释正确;stereotyping/explicit 成立;target(女性)未填,记召回缺失。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: 遇见4个<em class="keyword">母夜叉</em>
- **Transcript**: 兄弟们，今天出两件大事，气死我了，看等我买了一车哥伦比亚了吧，我准备去芝加哥途经这个奥特莱斯，我就进去了，买了这么多的哥伦比亚结账的时候呢，我付现金，他说他没有零钱，他找不了我，要我刷卡，我说我卡里没钱，我只付现金，要不然我不买了，四个大老娘们瞪着我给我堵那地方了，说L kiding me，你在耍我吗？我耍你老舅，我我怎么的付现金也不行吗？后来旁边有个白人的贵妇啊，就给我说两句好话啊，这事就。拉倒了。第二件事不回到车上了吗？我拿第二趟，因为太多了，装满满凳登的这地方呢就有一个人给他系鞋带。我关上车门之后，我就来个性感的锁车，这声呢确实有点大，大家挺一停啊。就给蹲着这人那下狱拘令，然后你要我来了，说你为什么吓唬我，我都蒙圈了，我锁车也就吓唬他。这老美有的时候不是误。😡
- **档案**: targets=[] | mechanism=["stereotyping"] | explicitness=explicit
- **modality_cues**: {"speech": "The speaker describes encountering four women they refer to as '母夜叉' (a derogatory term for women), and later mentions a white woman who speaks to them."}
- **neutral_summary**: The video features a person walking outdoors while speaking about their experience with four women they call '母夜叉', which is a derogatory term. They also mention a white woman who speaks to them. The video includes a transcript discussing the encounter.

#### `BV1jz4y1P77v`  — Offensive / harmful / split=val

- **判定: `faithful`** — 妈宝男相亲讽刺短剧,summary 对'听妈妈的话挑对象'内容转述准确,stereotyping/implicit 成立。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: 宁愿单身也不嫁给<em class="keyword">妈宝男</em>
- **Transcript**: 🎼你好，很高兴玩你主，我是貂蝉哦，好漂亮啊。😊对我还特意就是为您烫了一个卷发啊，就给我换一个啊，为什么就是卷发的女生，人家说都太浪了，我hold不住的。好吧，给我换一个我喜欢直发的女生，你白烫了。🎼您好，很高兴为您服务，我是您家的法师吗？啊，是的，这么好看，谢谢啊，这个帮我换一个，为为什么要换来啊？因为我妈妈说了，要温度，不要温度的女生不能要。今天外面天气冷，你才穿一件短袖就过来啊。把外套去穿上啊你好，我是小乔。这样元旦到乐都要勇往直前。哇，这么好看的，穿上给我换一个。😊🎼为什么我妈妈说了，相亲对象不能找这么漂亮的，就是尤其是打了耳洞的，hold不住的坏的。
- **档案**: targets=[] | mechanism=["stereotyping"] | explicitness=implicit
- **modality_cues**: {"speech": "The speaker expresses preferences about hair styles and相亲对象 (potential marriage partners)."}
- **neutral_summary**: A person discusses their preference for a straight-haired girl over a curly-haired one, mentioning their mother's advice about temperature and the weather being cold outside.

#### `BV1kT411m7Lm`  — Offensive / harmful / split=test

- **判定: `faithful`** — 针对个人(古振兴)的嘲讽'纪录片',cue 引用属实;mechanism=stereotyping 不贴切(更像 insult/mockery),记字段弱依据;标题 '母牛' 的辱骂未被点出。
- target 有依据: n.a. | mechanism 相符: no | 幻觉: no | 洗白: no
- **Title**: <em class="keyword">母牛</em>
- **Transcript**: 🎼给他一台电脑，他能连你的细胞都潮，他就是大名鼎鼎的古振星。当别的人还在为工作而奋斗时，他就已经学会了抄袭，开发了一个垃圾游戏迷你世界。就连蓝色妖姬也表示，我本以为我的齐尔西已经够厉害了。🎼但当我看见古振兴拿起他的电脑时，我就知道一切都是徒劳。欢迎您持续收看古振兴抄袭纪录片。
- **档案**: targets=[] | mechanism=["stereotyping"] | explicitness=implicit
- **modality_cues**: {"speech": "The video mentions '古振兴' (Gu Zhenxing) and his alleged copying of '蓝色妖姬' (Blue Witch) and '迷你世界' (Mini World).", "on_screen_text": "给他一台电脑，他能连你的细胞都潮，他就是大名鼎鼎的古振星。当别的人还在为工作而奋斗时，他就已经学会了抄袭，开发了一个垃圾游戏迷你世界。就连蓝色妖姬也表示，我本以为我的齐尔西已经够厉害了。"}
- **neutral_summary**: The video discusses Gu Zhenxing's alleged copying of Blue Witch and Mini World, suggesting he is a notorious figure. It implies that he has already copied before others have even started working.

#### `BV1ru411E7Ph`  — Offensive / harmful / split=train

- **判定: `whitewash`** — 标题'破鞋'是对(酒吧门口)女性的明确 slur 双关,视频以此贬损物化女性;档案 visual cue 却写'no explicit targeting or disparagement is observed',summary 说成'fashion and style'——毒性核心被洗掉(stereotyping/implicit 虽有 fire,但 summary+cue 主动否认贬损)。
- target 有依据: n.a. | mechanism 相符: no | 幻觉: no | 洗白: yes
- **Title**: 盘点酒吧门口小姐姐，那些我们穿不起的<em class="keyword">破鞋</em>
- **Transcript**: 🎼盘点酒吧小姐姐，那些我们穿不起的穿搭，一号姐姐穿的阿里世家弹簧鞋582张，咱找5张安排给您。😡2号姐姐穿的gucci老爹鞋，萌物84张，再加3张都划算的嘞。🎼最后妹子穿的巴黎是加3XL某物73张，咱家两张多搞定。穿。😊
- **档案**: targets=[] | mechanism=["stereotyping"] | explicitness=implicit
- **modality_cues**: {"visual": "The video shows women wearing high-heeled shoes and casual footwear, but no explicit targeting or disparagement is observed.", "on_screen_text": "The on-screen text includes brand names like '阿里世家' (Ali世家) and 'GUCCI', which are used to describe the shoes worn by the women."}
- **neutral_summary**: The video features women walking outdoors and using escalators, showcasing their outfits and footwear. The on-screen text mentions brands and shoe types, suggesting a focus on fashion and style.

#### `BV1834y1T71Z`  — Normal / normal / split=train

- **判定: `hallucination`** — 中医治阳痿偏方视频(benign 医疗内容),summary 忠实;mechanism=['stereotyping'] 无依据,字段级虚报(低危害)。
- target 有依据: n.a. | mechanism 相符: no | 幻觉: yes (field-level) | 洗白: no
- **Title**: 生姜加一物让<em class="keyword">阴茎</em>勃起，硬如铁，在家就能治阳痿
- **Transcript**: 生姜加衣物让阴茎勃起硬如铁。在家就能治疗阳痿。今天分享一个小妙方，让你自己在家就能改善阳痿勃起硬度差啊，总是时好时坏，中途疲软甚至无法勃起的情况。肾阳的推动对于迅速切持久的勃起至关重要，但肾阳不足，阴茎得不到足够的血液啊供应，就容易引起勃起困难和硬度差。准备一些生姜和赤勺切成片，放入开水中，浸泡5至10分钟，过滤后使用。生姜具有解表散寒的作用，能够补充阳气，驱散寒气，赤勺活去化瘀，使你的勃给迅速而持久。
- **档案**: targets=[] | mechanism=["stereotyping"] | explicitness=implicit
- **modality_cues**: {"speech": "The speaker discusses a remedy involving ginger and other ingredients to improve erectile function.", "on_screen_text": "The video title and on-screen text suggest a remedy for erectile dysfunction using ginger and other unspecified ingredients."}
- **neutral_summary**: A person in a lab coat discusses a remedy for erectile dysfunction, mentioning the importance of kidney yang and the use of ginger and other ingredients. The video includes on-screen text promoting the remedy.

#### `BV18w411o7SQ`  — Normal / normal / split=train

- **判定: `faithful`** — 键盘测评;mechanism=['slur']+explicit 触发自标题'基佬紫'——该词存在(词面依据),但它是约定俗成的颜色名,内容层面无攻击;记'词面触发/语境误判'而非虚构。与记忆编辑 (a) 的切片语义直接相关:LGBT 词面命中 ≠ 仇恨内容。
- target 有依据: n.a. | mechanism 相符: no | 幻觉: no | 洗白: no
- **Title**: 基佬紫硅胶键盘 试打
- **Transcript**: 这个这个还是有点用用不惯啊，就是你必须要一定要把这个轴，就是一定要把它的脚腕按到底层触发，有点难受。而且它大件不是整个大件里面都是胶腕啊，只有。😡他的这两边是有胶腕的，所以说要摁两边才能触发。然后这挺难用的。.静音都是很静啊，特别是这个这两个静音空格。
- **档案**: targets=[] | mechanism=["slur"] | explicitness=explicit
- **modality_cues**: {"visual": "A person is typing on a purple keyboard.", "speech": "The speaker expresses dissatisfaction with the keyboard's functionality, mentioning that the keys require a specific action to trigger and are difficult to use."}
- **neutral_summary**: The video shows a person typing on a purple keyboard while discussing its functionality. The speaker expresses frustration with the keyboard's design, noting that certain keys require a specific action to activate.

#### `BV1FN41167Ei`  — Normal / normal / split=train

- **判定: `hallucination`** — 医学科普(阴茎异常勃起),summary 忠实;mechanism=['stereotyping'] 无依据,字段级虚报(低危害)。ZH Normal 层的'涉性医疗内容→spurious mechanism'系统性模式之一。
- target 有依据: n.a. | mechanism 相符: no | 幻觉: yes (field-level) | 洗白: no
- **Title**: 《<em class="keyword">阴茎</em>异常勃起》
- **Transcript**: 正常成年男性在性活动或持续性刺激下，阴茎勃起能持续数分钟甚至1小时以上。若在非上状态下，阴茎勃起持续超过4小时，医学上称为阴茎异常勃起。🎼其病因包括外伤、高度饮酒、暴力行为、血液疾病、轴瘤区范、手术损伤等等，特别是异常勃起超过6小时就会有阴茎坏死的风险，所以及时就医为上策，以免造成更大的终身伤害。
- **档案**: targets=[] | mechanism=["stereotyping"] | explicitness=none
- **modality_cues**: {"visual": "The speaker is wearing a dark blue shirt and glasses, sitting at a desk with a computer keyboard.", "speech": "The video discusses abnormal penile erections and their potential risks.", "on_screen_text": "The video title mentions 'abnormal penile erection' and provides medical information about its causes and risks."}
- **neutral_summary**: The video features a medical professional discussing the topic of abnormal penile erections, explaining the conditions under which such erections occur and the associated risks, including the possibility of tissue damage if left untreated.

#### `BV1GL411q7rm`  — Normal / normal / split=train

- **判定: `hallucination`** — 小孩跳'挖呀挖'儿歌,summary 忠实;mechanism=['stereotyping'] 无依据,字段级虚报(低危害)。
- target 有依据: n.a. | mechanism 相符: no | 幻觉: yes (field-level) | 洗白: no
- **Title**: 学生日常
- **Transcript**: 🎼1234在小小的花园里面挖呀挖呀挖种小小的种子开小小的花。在大大的花园里面挖呀挖呀挖，种大大的种子开大大的花，在特别大的花园里面挖呀挖呀挖，种特别大的种子开特别大的花，我园小小的花，园出小小的花。🎼期待和你一起合。😊
- **档案**: targets=[] | mechanism=["stereotyping"] | explicitness=none
- **modality_cues**: {"visual": "A child wearing glasses and a t-shirt with 'I Love Dad' printed on it is dancing in a room."}
- **neutral_summary**: The video shows a child dancing in a room while wearing glasses and a t-shirt with 'I Love Dad' printed on it. The child appears to be enjoying themselves.

#### `BV1Go4y1J7ZA`  — Normal / normal / split=train

- **判定: `faithful`** — benign 追星内容,档案一致且字段全空——正确的空输出。
- target 有依据: n.a. | mechanism 相符: n.a. | 幻觉: no | 洗白: no
- **Title**: 原来也有让杨幂一直犯花痴的人
- **Transcript**: 🎼让你继续加油，我会一直支持你是我的。😊
- **档案**: targets=[] | mechanism=[] | explicitness=none
- **modality_cues**: {"visual": "The video features a woman holding a bouquet of flowers and smiling at someone off-camera."}
- **neutral_summary**: The video shows a woman holding a bouquet of flowers and smiling, with a supportive message in Chinese text indicating ongoing support.

#### `BV1MU4y1D7Ks`  — Normal / normal / split=test

- **判定: `faithful`** — 转写明确贬低无性恋者('这种人最可怕了'、'没有正常人的欲望'),targets=LGBTQ+、stereotyping+dehumanization 均有转写依据。注意:gt 标签是 Normal——这是与 W2 两例方向相反的'档案报毒/标签说无害'分歧,疑似漏标的标签噪声(test split),建议人工复核。
- target 有依据: yes | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: 你知道无性恋吗？
- **Transcript**: 🎼你知道无性恋吗？无性恋，他身上没有正常人都会表现出来的那种欲望。这种人最可怕了。我说的没错，在他身上，我看不到任何的口效，明明连神性都会有七情六欲的。
- **档案**: targets=["LGBTQ+"] | mechanism=["stereotyping", "dehumanization"] | explicitness=explicit
- **modality_cues**: {"visual": "A person in a white outfit is walking down a city street.", "speech": "你知道无性恋吗？无性恋，他身上没有正常人都会表现出来的那种欲望。这种人最可怕了。我说的没错，在他身上，我看不到任何的口效，明明连神性都会有七情六欲的。"}
- **neutral_summary**: The video features a person walking down a city street while discussing LGBTQ+ individuals, using derogatory terms and stereotypes.

#### `BV1N8411R7h7`  — Normal / normal / split=train

- **判定: `faithful`** — '妈宝男'自嘲段子,OCR 与标题+转写一致;stereotyping 有词面依据;summary 称 on-screen text 含 derogatory terms 略过重但非虚构。
- target 有依据: n.a. | mechanism 相符: yes | 幻觉: no | 洗白: no
- **Title**: <em class="keyword">妈宝男</em>～
- **Transcript**: 妈妈呢妈，你听见了吗？别侮辱我，我天天自己吃，自己喝，自己能上桌，每天自行睡觉8小时独立拉屎，还不用拿纸跟谁俩？😡
- **档案**: targets=[] | mechanism=["stereotyping"] | explicitness=explicit
- **modality_cues**: {"visual": "A person is seen in a yellow coat, and another person is sitting on a chair with a child. There is also a person in a red shirt standing outdoors.", "on_screen_text": "妈宝男～ . 妈妈呢妈，你听见了吗？别侮辱我，我天天自己吃，自己喝，自己能上桌，每天自行睡觉8小时独立拉屎，还不用拿纸跟谁俩？😡"}
- **neutral_summary**: The video features various scenes including a person in a yellow coat, a person sitting with a child, and a person in a red shirt standing outdoors. The on-screen text includes derogatory terms and phrases.

#### `BV1hr4y1T7VM`  — Normal / normal / split=train

- **判定: `hallucination`** — 妇产科医生科普阴超 vs 阴道镜,summary 忠实;mechanism=['stereotyping'] 无依据,字段级虚报(低危害)。
- target 有依据: n.a. | mechanism 相符: no | 幻觉: yes (field-level) | 洗白: no
- **Title**: 宫颈有病变，要做阴超还是阴道镜？
- **Transcript**: 그.大家好，我是恩哥，一名妇产科医生。今天有一位粉丝咨询说宫颈上有点小毛病，我了解病情之后，建议他进一步做个阴道镜评估。他说我做过啦，然后贴了一张阴超报告给我，我哭笑不得。阴道镜跟阴超完全两个东西，不是一回事。阴道镜类似一个放大镜的装置。如果阴道壁或者宫颈表面怀疑有病变，可以在阴道镜的帮助下，找到可疑的点，然后取活组织送病理检查，但是他看不了子宫卵巢里面的问题。而阴超有这个功能，阴超是经阴道。做的B超，所以简称阴超，可以看到子宫宫腔内膜卵巢有什么结构异常或者肿块，宫颈上的肿块或者囊肿、阴道壁的囊肿也一样看得出，但是他恰恰看不了阴道壁跟宫颈表面的问题。更多妇产科知识，欢迎关注恩哥聊健康，谢谢大家。
- **档案**: targets=[] | mechanism=["stereotyping"] | explicitness=none
- **modality_cues**: {"speech": "The speaker discusses the differences between a vaginal ultrasound (阴超) and a colposcopy (阴道镜) for diagnosing cervical abnormalities."}
- **neutral_summary**: A medical professional explains the distinctions between a vaginal ultrasound and a colposcopy for evaluating cervical abnormalities.

#### `BV1pK411G7vj`  — Normal / normal / split=train

- **判定: `hallucination`** — 两人看帅哥的'花痴日常',cue 引用属实、summary 忠实;mechanism=['insult']+explicit 无依据('我操'是感叹词,无受辱对象),字段级虚报(低危害)。
- target 有依据: n.a. | mechanism 相符: no | 幻觉: yes (field-level) | 洗白: no
- **Title**: 花痴日常
- **Transcript**: 🎼看我这是帅哥，哎呦什么呀？看美男美男，看电视了，我操。远。🎼却日就是千千晚星，亮过今晚月亮。😊
- **档案**: targets=[] | mechanism=["insult"] | explicitness=explicit
- **modality_cues**: {"visual": "Two women are seen in a room, one sitting in a chair and the other standing. They appear to be engaged in a conversation.", "speech": "The transcript includes phrases like '帅哥' (handsome guy) and '美男' (beautiful man).", "on_screen_text": "The on-screen text includes '这什么呀～看美男!' which translates to 'What's this? Watching handsome men!'"}
- **neutral_summary**: The video depicts two women in a room, one sitting and the other standing, engaging in a conversation. The transcript contains phrases that suggest admiration for attractive individuals, specifically men.

#### `BV1xN411u7Jh`  — Normal / normal / split=train

- **判定: `hallucination`** — 新年祝福直播片段,cue 自己解释'舰长'是尊称、summary 完全 benign,却输出 mechanism=['slur']——自相矛盾的字段级虚报。
- target 有依据: n.a. | mechanism 相符: no | 幻觉: yes (field-level) | 洗白: no
- **Title**: hanser当着大伟哥的面祝各位舰长鸡年大吉吧
- **Transcript**: 🎼我们把现场交回给官方的直播间，顺带祝各位舰长话，祝各位舰长新年大吉吧。你只是想说。8。7掉了吗？7掉了吗？好，了吗？那个我们我们直播我我们直播屏幕又回到我们现场。那么我们感谢我们的那个非感谢我们hansa和尤克里的为我们的这个这个非常非常非常有意思的这个演绎。😊
- **档案**: targets=[] | mechanism=["slur"] | explicitness=none
- **modality_cues**: {"speech": "The speaker uses the term '舰长' which is a respectful term for a ship's captain, likely referring to players in a game context."}
- **neutral_summary**: The video features two individuals seated at a table with laptops, engaging in a discussion. The setting includes a backdrop with animated characters, and the screen displays a game interface with various icons and a character model. The transcript includes phrases like '鸡年大吉' (Happy Rooster Year) and '新年大吉' (Happy New Year), indicating a celebratory tone.

