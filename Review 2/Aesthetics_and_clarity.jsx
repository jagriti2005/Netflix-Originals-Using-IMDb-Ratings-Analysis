import React, { useState, useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, ScatterChart, Scatter, PieChart, Pie, Cell, AreaChart, Area, RadialBarChart, RadialBar } from 'recharts';
import { TrendingUp, Film, Star, Calendar, BarChart3, Target, Award, Palette, Eye } from 'lucide-react';

const NetflixAestheticsAnalysis = () => {
  const [activeVisualization, setActiveVisualization] = useState('elegant-bars');
  const [isDarkMode, setIsDarkMode] = useState(false);

  // Sample Netflix data
  const netflixData = [
    { title: "Dark", year: 2017, genre: "Sci-Fi", imdb: 8.8, type: "Series", popularity: 95, runtime: 60 },
    { title: "Narcos", year: 2015, genre: "Crime", imdb: 8.8, type: "Series", popularity: 92, runtime: 50 },
    { title: "Stranger Things", year: 2016, genre: "Sci-Fi", imdb: 8.7, type: "Series", popularity: 98, runtime: 50 },
    { title: "The Crown", year: 2016, genre: "Drama", imdb: 8.6, type: "Series", popularity: 88, runtime: 60 },
    { title: "House of Cards", year: 2013, genre: "Drama", imdb: 8.6, type: "Series", popularity: 85, runtime: 50 },
    { title: "Ozark", year: 2017, genre: "Crime", imdb: 8.4, type: "Series", popularity: 89, runtime: 60 },
    { title: "Money Heist", year: 2017, genre: "Crime", imdb: 8.3, type: "Series", popularity: 94, runtime: 70 },
    { title: "The Witcher", year: 2019, genre: "Fantasy", imdb: 8.2, type: "Series", popularity: 91, runtime: 60 },
    { title: "Orange Is the New Black", year: 2013, genre: "Drama", imdb: 8.1, type: "Series", popularity: 82, runtime: 60 },
    { title: "Marriage Story", year: 2019, genre: "Drama", imdb: 7.9, type: "Movie", popularity: 76, runtime: 137 },
    { title: "The Irishman", year: 2019, genre: "Crime", imdb: 7.8, type: "Movie", popularity: 79, runtime: 209 },
    { title: "Roma", year: 2018, genre: "Drama", imdb: 7.7, type: "Movie", popularity: 74, runtime: 135 },
    { title: "Bridgerton", year: 2020, genre: "Romance", imdb: 7.3, type: "Series", popularity: 87, runtime: 55 },
    { title: "Extraction", year: 2020, genre: "Action", imdb: 6.7, type: "Movie", popularity: 83, runtime: 116 },
    { title: "Bird Box", year: 2018, genre: "Horror", imdb: 6.6, type: "Movie", popularity: 90, runtime: 124 },
    { title: "Enola Holmes", year: 2020, genre: "Mystery", imdb: 6.6, type: "Movie", popularity: 78, runtime: 123 },
  ];

  // Enhanced color palettes for better aesthetics
  const colorPalettes = {
    netflix: ['#E50914', '#221F1F', '#F5F5F1', '#564D4D', '#831010'],
    modern: ['#667EEA', '#764BA2', '#F093FB', '#F5576C', '#4FACFE'],
    gradient: ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
    elegant: ['#2C3E50', '#3498DB', '#E74C3C', '#F39C12', '#27AE60'],
    pastel: ['#A8E6CF', '#DCEDC8', '#FFD3A5', '#FD99A8', '#CDBADB']
  };

  const currentPalette = isDarkMode ? colorPalettes.modern : colorPalettes.elegant;

  // Processed data for different visualizations
  const topRatedShows = netflixData
    .sort((a, b) => b.imdb - a.imdb)
    .slice(0, 8)
    .map(item => ({
      ...item,
      shortTitle: item.title.length > 15 ? item.title.substring(0, 15) + '...' : item.title
    }));

  const genreData = useMemo(() => {
    const genres = {};
    netflixData.forEach(item => {
      if (!genres[item.genre]) {
        genres[item.genre] = { genre: item.genre, count: 0, totalRating: 0, avgRating: 0 };
      }
      genres[item.genre].count++;
      genres[item.genre].totalRating += item.imdb;
    });

    return Object.values(genres)
      .map(g => ({
        ...g,
        avgRating: parseFloat((g.totalRating / g.count).toFixed(1))
      }))
      .sort((a, b) => b.avgRating - a.avgRating);
  }, []);

  const yearlyData = useMemo(() => {
    const years = {};
    netflixData.forEach(item => {
      if (!years[item.year]) {
        years[item.year] = { year: item.year, count: 0, totalRating: 0, avgRating: 0 };
      }
      years[item.year].count++;
      years[item.year].totalRating += item.imdb;
    });

    return Object.values(years)
      .map(y => ({
        ...y,
        avgRating: parseFloat((y.totalRating / y.count).toFixed(1))
      }))
      .sort((a, b) => a.year - b.year);
  }, []);

  const popularityRatingData = netflixData.map(item => ({
    popularity: item.popularity,
    imdb: item.imdb,
    title: item.title,
    genre: item.genre,
    size: item.type === 'Series' ? 100 : 60
  }));

  // Custom styled components for better aesthetics
  const CustomTooltip = ({ active, payload, label, isDark = false }) => {
    if (active && payload && payload.length) {
      return (
        <div className={`p-4 rounded-xl border shadow-2xl backdrop-blur-sm ${
          isDark 
            ? 'bg-gray-800/90 border-gray-600 text-white' 
            : 'bg-white/90 border-gray-200 text-gray-800'
        }`}>
          <p className="font-bold text-lg mb-2">{label}</p>
          {payload.map((entry, index) => (
            <div key={index} className="flex items-center gap-2 mb-1">
              <div 
                className="w-3 h-3 rounded-full" 
                style={{ backgroundColor: entry.color }}
              />
              <span className="font-medium">{entry.name}:</span>
              <span className="font-bold">{entry.value}</span>
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  const ScatterTooltip = ({ active, payload, isDark = false }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className={`p-4 rounded-xl border shadow-2xl backdrop-blur-sm ${
          isDark 
            ? 'bg-gray-800/90 border-gray-600 text-white' 
            : 'bg-white/90 border-gray-200 text-gray-800'
        }`}>
          <p className="font-bold text-lg mb-2">{data.title}</p>
          <div className="space-y-1">
            <p><span className="font-medium">IMDb Rating:</span> <span className="font-bold text-yellow-500">{data.imdb}</span></p>
            <p><span className="font-medium">Popularity:</span> <span className="font-bold text-blue-500">{data.popularity}%</span></p>
            <p><span className="font-medium">Genre:</span> <span className="font-bold">{data.genre}</span></p>
          </div>
        </div>
      );
    }
    return null;
  };

  const visualizations = [
    {
      id: 'elegant-bars',
      title: 'Top Rated Netflix Originals',
      subtitle: 'Elegant horizontal bar chart with refined styling',
      icon: <BarChart3 className="w-6 h-6" />
    },
    {
      id: 'gradient-genre',
      title: 'Genre Performance Analysis',
      subtitle: 'Gradient-filled bars with modern color scheme',
      icon: <Film className="w-6 h-6" />
    },
    {
      id: 'smooth-trends',
      title: 'Quality Trends Over Time',
      subtitle: 'Smooth curved lines with enhanced visual appeal',
      icon: <TrendingUp className="w-6 h-6" />
    },
    {
      id: 'artistic-scatter',
      title: 'Popularity vs Quality Matrix',
      subtitle: 'Artistic scatter plot with size-coded bubbles',
      icon: <Target className="w-6 h-6" />
    },
    {
      id: 'radial-ratings',
      title: 'Rating Distribution Wheel',
      subtitle: 'Radial bar chart for unique perspective',
      icon: <Award className="w-6 h-6" />
    }
  ];

  const renderVisualization = () => {
    const bgColor = isDarkMode ? '#1F2937' : '#FFFFFF';
    const gridColor = isDarkMode ? '#374151' : '#E5E7EB';
    const textColor = isDarkMode ? '#F9FAFB' : '#1F2937';

    switch (activeVisualization) {
      case 'elegant-bars':
        return (
          <ResponsiveContainer width="100%" height={500}>
            <BarChart 
              data={topRatedShows} 
              layout="horizontal"
              margin={{ top: 20, right: 30, left: 80, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke={gridColor} strokeOpacity={0.3} />
              <XAxis 
                type="number" 
                domain={[6, 9]} 
                tick={{ fill: textColor, fontSize: 12, fontWeight: 500 }}
                axisLine={{ stroke: gridColor }}
              />
              <YAxis 
                type="category" 
                dataKey="shortTitle" 
                tick={{ fill: textColor, fontSize: 12, fontWeight: 500 }}
                axisLine={{ stroke: gridColor }}
                width={70}
              />
              <Tooltip content={<CustomTooltip isDark={isDarkMode} />} />
              <Bar 
                dataKey="imdb" 
                fill="url(#elegantGradient)"
                radius={[0, 8, 8, 0]}
                stroke={currentPalette[0]}
                strokeWidth={1}
              />
              <defs>
                <linearGradient id="elegantGradient" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor={currentPalette[0]} stopOpacity={0.8}/>
                  <stop offset="100%" stopColor={currentPalette[1]} stopOpacity={0.9}/>
                </linearGradient>
              </defs>
            </BarChart>
          </ResponsiveContainer>
        );

      case 'gradient-genre':
        return (
          <ResponsiveContainer width="100%" height={500}>
            <BarChart data={genreData} margin={{ top: 20, right: 30, left: 20, bottom: 80 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridColor} strokeOpacity={0.3} />
              <XAxis 
                dataKey="genre" 
                angle={-45} 
                textAnchor="end" 
                height={80}
                tick={{ fill: textColor, fontSize: 11, fontWeight: 500 }}
                axisLine={{ stroke: gridColor }}
              />
              <YAxis 
                domain={[0, 10]} 
                tick={{ fill: textColor, fontSize: 12, fontWeight: 500 }}
                axisLine={{ stroke: gridColor }}
              />
              <Tooltip content={<CustomTooltip isDark={isDarkMode} />} />
              <Bar dataKey="avgRating" radius={[8, 8, 0, 0]} stroke="#fff" strokeWidth={2}>
                {genreData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={currentPalette[index % currentPalette.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        );

      case 'smooth-trends':
        return (
          <ResponsiveContainer width="100%" height={500}>
            <AreaChart data={yearlyData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridColor} strokeOpacity={0.3} />
              <XAxis 
                dataKey="year" 
                tick={{ fill: textColor, fontSize: 12, fontWeight: 500 }}
                axisLine={{ stroke: gridColor }}
              />
              <YAxis 
                domain={[7, 9]} 
                tick={{ fill: textColor, fontSize: 12, fontWeight: 500 }}
                axisLine={{ stroke: gridColor }}
              />
              <Tooltip content={<CustomTooltip isDark={isDarkMode} />} />
              <defs>
                <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={currentPalette[0]} stopOpacity={0.8}/>
                  <stop offset="100%" stopColor={currentPalette[0]} stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <Area 
                type="monotone" 
                dataKey="avgRating" 
                stroke={currentPalette[0]} 
                strokeWidth={4}
                fill="url(#areaGradient)"
                dot={{ fill: currentPalette[1], strokeWidth: 2, r: 6 }}
                activeDot={{ r: 8, stroke: currentPalette[0], strokeWidth: 2, fill: '#fff' }}
              />
            </AreaChart>
          </ResponsiveContainer>
        );

      case 'artistic-scatter':
        return (
          <ResponsiveContainer width="100%" height={500}>
            <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridColor} strokeOpacity={0.3} />
              <XAxis 
                type="number" 
                dataKey="popularity" 
                name="Popularity %" 
                domain={[70, 100]}
                tick={{ fill: textColor, fontSize: 12, fontWeight: 500 }}
                axisLine={{ stroke: gridColor }}
              />
              <YAxis 
                type="number" 
                dataKey="imdb" 
                name="IMDb Rating" 
                domain={[6, 9]}
                tick={{ fill: textColor, fontSize: 12, fontWeight: 500 }}
                axisLine={{ stroke: gridColor }}
              />
              <Tooltip content={<ScatterTooltip isDark={isDarkMode} />} />
              <Scatter data={popularityRatingData} fill={currentPalette[0]}>
                {popularityRatingData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={currentPalette[index % currentPalette.length]}
                    fillOpacity={0.8}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        );

      case 'radial-ratings':
        const ratingDistribution = [
          { name: '6.0-7.0', value: 4, fill: currentPalette[0] },
          { name: '7.0-8.0', value: 6, fill: currentPalette[1] },
          { name: '8.0-9.0', value: 6, fill: currentPalette[2] }
        ];
        
        return (
          <ResponsiveContainer width="100%" height={500}>
            <RadialBarChart cx="50%" cy="50%" innerRadius="30%" outerRadius="80%" data={ratingDistribution}>
              <RadialBar 
                minAngle={15} 
                label={{ position: 'insideStart', fill: '#fff', fontWeight: 'bold' }} 
                background 
                clockWise 
                dataKey="value" 
                cornerRadius={10}
              />
              <Tooltip content={<CustomTooltip isDark={isDarkMode} />} />
            </RadialBarChart>
          </ResponsiveContainer>
        );

      default:
        return null;
    }
  };

  const currentViz = visualizations.find(v => v.id === activeVisualization);

  return (
    <div className={`min-h-screen transition-all duration-500 ${
      isDarkMode 
        ? 'bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900' 
        : 'bg-gradient-to-br from-slate-50 via-white to-blue-50'
    }`}>
      <div className="p-8">
        <div className="max-w-7xl mx-auto">
          {/* Header with Dark Mode Toggle */}
          <div className="flex justify-between items-center mb-8">
            <div className="text-center flex-1">
              <h1 className={`text-5xl font-black mb-3 ${
                isDarkMode ? 'text-white' : 'text-gray-900'
              }`}>
                Netflix Originals
              </h1>
              <p className={`text-xl ${
                isDarkMode ? 'text-gray-300' : 'text-gray-600'
              }`}>
                Aesthetics & Clarity in Data Visualization
              </p>
            </div>
            <button
              onClick={() => setIsDarkMode(!isDarkMode)}
              className={`p-3 rounded-xl transition-all duration-300 ${
                isDarkMode 
                  ? 'bg-gray-700 hover:bg-gray-600 text-yellow-400' 
                  : 'bg-gray-100 hover:bg-gray-200 text-gray-600'
              }`}
            >
              {isDarkMode ? '☀️' : '🌙'}
            </button>
          </div>

          {/* Visualization Selector */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
            {visualizations.map((viz) => (
              <button
                key={viz.id}
                onClick={() => setActiveVisualization(viz.id)}
                className={`p-6 rounded-2xl border-2 transition-all duration-300 text-left group ${
                  activeVisualization === viz.id
                    ? isDarkMode
                      ? 'border-blue-400 bg-blue-900/30 shadow-2xl shadow-blue-500/20'
                      : 'border-blue-500 bg-blue-50 shadow-2xl shadow-blue-500/20'
                    : isDarkMode
                      ? 'border-gray-600 bg-gray-800/50 hover:border-gray-500 hover:shadow-lg'
                      : 'border-gray-200 bg-white/70 hover:border-gray-300 hover:shadow-lg backdrop-blur-sm'
                }`}
              >
                <div className="flex items-center gap-3 mb-3">
                  <div className={`p-2 rounded-lg ${
                    activeVisualization === viz.id
                      ? isDarkMode ? 'bg-blue-500' : 'bg-blue-500'
                      : isDarkMode ? 'bg-gray-700' : 'bg-gray-100'
                  }`}>
                    <div className={activeVisualization === viz.id ? 'text-white' : isDarkMode ? 'text-gray-300' : 'text-gray-600'}>
                      {viz.icon}
                    </div>
                  </div>
                  <h3 className={`font-bold ${
                    isDarkMode ? 'text-white' : 'text-gray-900'
                  }`}>
                    {viz.title}
                  </h3>
                </div>
                <p className={`text-sm ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  {viz.subtitle}
                </p>
              </button>
            ))}
          </div>

          {/* Main Visualization Card */}
          <div className={`rounded-3xl shadow-2xl backdrop-blur-sm border transition-all duration-500 ${
            isDarkMode 
              ? 'bg-gray-800/70 border-gray-600' 
              : 'bg-white/80 border-gray-200'
          }`}>
            <div className="p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className={`p-3 rounded-xl ${
                  isDarkMode ? 'bg-blue-600' : 'bg-blue-500'
                }`}>
                  <div className="text-white">
                    {currentViz?.icon}
                  </div>
                </div>
                <div>
                  <h2 className={`text-3xl font-bold mb-1 ${
                    isDarkMode ? 'text-white' : 'text-gray-900'
                  }`}>
                    {currentViz?.title}
                  </h2>
                  <p className={`${
                    isDarkMode ? 'text-gray-400' : 'text-gray-600'
                  }`}>
                    {currentViz?.subtitle}
                  </p>
                </div>
              </div>
              
              {renderVisualization()}
            </div>
          </div>

          {/* Aesthetic Principles */}
          <div className={`mt-8 rounded-3xl shadow-xl backdrop-blur-sm border p-8 ${
            isDarkMode 
              ? 'bg-gray-800/70 border-gray-600' 
              : 'bg-white/80 border-gray-200'
          }`}>
            <div className="flex items-center gap-3 mb-6">
              <Palette className={`w-8 h-8 ${isDarkMode ? 'text-purple-400' : 'text-purple-600'}`} />
              <h3 className={`text-2xl font-bold ${
                isDarkMode ? 'text-white' : 'text-gray-900'
              }`}>
                Aesthetic Design Principles Applied
              </h3>
            </div>
            
            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <h4 className={`text-lg font-semibold mb-3 flex items-center gap-2 ${
                  isDarkMode ? 'text-blue-400' : 'text-blue-600'
                }`}>
                  <Eye className="w-5 h-5" />
                  Visual Hierarchy
                </h4>
                <ul className={`space-y-2 ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  <li>• Bold, readable typography with clear size hierarchy</li>
                  <li>• Strategic use of color to guide attention</li>
                  <li>• Consistent spacing and alignment</li>
                  <li>• Interactive states with smooth transitions</li>
                </ul>
              </div>
              
              <div>
                <h4 className={`text-lg font-semibold mb-3 flex items-center gap-2 ${
                  isDarkMode ? 'text-green-400' : 'text-green-600'
                }`}>
                  <Star className="w-5 h-5" />
                  Enhanced Clarity
                </h4>
                <ul className={`space-y-2 ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  <li>• Custom tooltips with rich information</li>
                  <li>• Rounded corners and subtle shadows for depth</li>
                  <li>• Gradient fills and stroke enhancements</li>
                  <li>• Responsive design for all screen sizes</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NetflixAestheticsAnalysis;