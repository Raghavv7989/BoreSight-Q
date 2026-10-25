/**
 * Pre-calibrated demo agricultural parcels in hard-rock granite/gneiss terrain
 * with high hydrogeological fracture complexity.
 */

export const DEMO_PARCELS = [
  {
    name: "Chikkaballapur Granite Gneiss Basin (Karnataka)",
    description: "Hard-rock granitic aquifer with deep orthogonal conjugate shear fractures.",
    center: [77.7275, 13.4350],
    zoom: 14.5,
    polygon: {
      type: "Polygon",
      coordinates: [
        [
          [77.7215, 13.4300],
          [77.7345, 13.4290],
          [77.7360, 13.4395],
          [77.7240, 13.4410],
          [77.7215, 13.4300],
        ]
      ]
    }
  },
  {
    name: "Anantapur Drought-Prone Agro-Tract (Rayalaseema)",
    description: "Semi-arid basaltic/gneiss terrain with critical micro-drainage convergence.",
    center: [77.5920, 14.6810],
    zoom: 14.8,
    polygon: {
      type: "Polygon",
      coordinates: [
        [
          [77.5860, 14.6765],
          [77.5985, 14.6750],
          [77.6000, 14.6860],
          [77.5880, 14.6875],
          [77.5860, 14.6765],
        ]
      ]
    }
  },
  {
    name: "Sahara Desert - Murzuq Erg (Libya)",
    description: "Hyper-arid desert sand sea with extreme aridity (<15 mm/yr precipitation) and dry sand dielectric backscatter.",
    center: [13.5000, 25.5000],
    zoom: 14.5,
    polygon: {
      type: "Polygon",
      coordinates: [
        [
          [13.4920, 25.4950],
          [13.5080, 25.4950],
          [13.5080, 25.5050],
          [13.4920, 25.5050],
          [13.4920, 25.4950],
        ]
      ]
    }
  },
  {
    name: "Arabian Sea (Open Marine Water)",
    description: "Deep offshore marine surface water. Absolute NDWI > 0.85, zero borewell drilling applicability.",
    center: [72.0000, 15.0000],
    zoom: 13.5,
    polygon: {
      type: "Polygon",
      coordinates: [
        [
          [71.9900, 14.9900],
          [72.0100, 14.9900],
          [72.0100, 15.0100],
          [71.9900, 15.0100],
          [71.9900, 14.9900],
        ]
      ]
    }
  }
];
