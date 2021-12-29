#include "huffmancodedpos.hpp"

namespace Shogi {

// Binary表記
// Binary<11110>::value とすれば、30 となる。
// 符合なし64bitなので19桁まで表記可能。
template<uint64_t n> struct Binary { static const uint64_t value = n % 10 + (Binary<n / 10>::value << 1); };
// template 特殊化
template<> struct Binary<0> { static const uint64_t value = 0; };

// clang-format off
const HuffmanCode HuffmanCodedPos::boardCodeTable[PieceNum] = {
    {Binary<         0>::value, 1}, // Empty
    {Binary<         1>::value, 4}, // BPawn
    {Binary<        11>::value, 6}, // BLance
    {Binary<       111>::value, 6}, // BKnight
    {Binary<      1011>::value, 6}, // BSilver
    {Binary<     11111>::value, 8}, // BBishop
    {Binary<    111111>::value, 8}, // BRook
    {Binary<      1111>::value, 6}, // BGold
    {Binary<         0>::value, 0}, // BKing 玉の位置は別途、位置を符号化する。使用しないので numOfBit を 0 にしておく。
    {Binary<      1001>::value, 4}, // BProPawn
    {Binary<    100011>::value, 6}, // BProLance
    {Binary<    100111>::value, 6}, // BProKnight
    {Binary<    101011>::value, 6}, // BProSilver
    {Binary<  10011111>::value, 8}, // BHorse
    {Binary<  10111111>::value, 8}, // BDragona
    {Binary<         0>::value, 0}, // 使用しないので numOfBit を 0 にしておく。
    {Binary<         0>::value, 0}, // 使用しないので numOfBit を 0 にしておく。
    {Binary<       101>::value, 4}, // WPawn
    {Binary<     10011>::value, 6}, // WLance
    {Binary<     10111>::value, 6}, // WKnight
    {Binary<     11011>::value, 6}, // WSilver
    {Binary<   1011111>::value, 8}, // WBishop
    {Binary<   1111111>::value, 8}, // WRook
    {Binary<    101111>::value, 6}, // WGold
    {Binary<         0>::value, 0}, // WKing 玉の位置は別途、位置を符号化する。
    {Binary<      1101>::value, 4}, // WProPawn
    {Binary<    110011>::value, 6}, // WProLance
    {Binary<    110111>::value, 6}, // WProKnight
    {Binary<    111011>::value, 6}, // WProSilver
    {Binary<  11011111>::value, 8}, // WHorse
    {Binary<  11111111>::value, 8}, // WDragon
};

// 盤上の bit 数 - 1 で表現出来るようにする。持ち駒があると、盤上には Empty の 1 bit が増えるので、
// これで局面の bit 数が固定化される。
const HuffmanCode HuffmanCodedPos::handCodeTable[HAND_PIECE_KIND_NUM][ColorNum] = {
    {{Binary<        0>::value, 3}, {Binary<      100>::value, 3}}, // HPawn
    {{Binary<        1>::value, 5}, {Binary<    10001>::value, 5}}, // HLance
    {{Binary<       11>::value, 5}, {Binary<    10011>::value, 5}}, // HKnight
    {{Binary<      101>::value, 5}, {Binary<    10101>::value, 5}}, // HSilver
    {{Binary<      111>::value, 5}, {Binary<    10111>::value, 5}}, // HGold
    {{Binary<    11111>::value, 7}, {Binary<  1011111>::value, 7}}, // HBishop
    {{Binary<   111111>::value, 7}, {Binary<  1111111>::value, 7}}, // HRook
};
// clang-format on

HuffmanCodeToPieceHash HuffmanCodedPos::boardCodeToPieceHash;
HuffmanCodeToPieceHash HuffmanCodedPos::handCodeToPieceHash;
}